#![crate_type = "cdylib"]

#[no_mangle]
pub extern fn func(x_ptr : *const f64, n : usize)
{
    println!("Hello, world! [from Rust]");

    let x = unsafe{ std::slice::from_raw_parts(x_ptr, n) };
    println!("{}/{}", x[0], n);
}
