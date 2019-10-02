#![crate_type = "cdylib"]

// vector add y = x + βy
#[no_mangle]
pub extern fn Add(y_ptr: *mut f64, x_ptr : *const f64, n : usize, beta : f64){
    let x = unsafe{ std::slice::from_raw_parts(x_ptr, n) };
    let y = unsafe{ std::slice::from_raw_parts_mut(y_ptr, n) };

    for i in 0..n {
        y[i] = x[i] + beta * y[i];
    }
}

// vector add y += αx
#[no_mangle]
pub extern fn AddSelf(y_ptr: *mut f64, x_ptr : *const f64, n : usize, alpha : f64){
    let x = unsafe{ std::slice::from_raw_parts(x_ptr, n) };
    let y = unsafe{ std::slice::from_raw_parts_mut(y_ptr, n) };

    for i in 0..n {
        y[i] = alpha * x[i];
    }
}