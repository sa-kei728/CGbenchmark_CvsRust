#![crate_type = "cdylib"]

// y = Ax
#[no_mangle]
pub extern fn SpMV(y_ptr: *mut f64, data_ptr : *const f64, column_ptr : *const usize, nonzero_ptr : *const usize,
                    x_ptr : *const f64, max_nonzero : usize, n : usize){
    let x = unsafe{ std::slice::from_raw_parts(x_ptr, n) };
    let data = unsafe{ std::slice::from_raw_parts(data_ptr, n * max_nonzero) };
    let column = unsafe{ std::slice::from_raw_parts(column_ptr, n * max_nonzero) };
    let nonzero = unsafe{ std::slice::from_raw_parts(nonzero_ptr, n) };
    let y = unsafe{ std::slice::from_raw_parts_mut(y_ptr, n) };

    for i in 0..n {
        let nnz = nonzero[i];
        let mut y_i = 0f64;
        for idx in 0..nnz {
            let a_ij = data[i * max_nonzero + idx];
            let j = column[i * max_nonzero + idx];
            let x_j = x[j];

            y_i += a_ij * x_j;
        }
        y[i] = y_i;
    }
}

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

// vector sub z = x - y
#[no_mangle]
pub extern fn Sub(z_ptr : *mut f64, x_ptr : *const f64, y_ptr: *const f64, n : usize){
    let x = unsafe{ std::slice::from_raw_parts(x_ptr, n) };
    let y = unsafe{ std::slice::from_raw_parts(y_ptr, n) };
    let z = unsafe{ std::slice::from_raw_parts_mut(z_ptr, n) };

    for i in 0..n {
        z[i] = x[i] - y[i];
    }
}

// vector dot r = x ・ y
#[no_mangle]
pub extern fn Dot(x_ptr : *const f64, y_ptr: *const f64, n : usize) -> f64 {
    let x = unsafe{ std::slice::from_raw_parts(x_ptr, n) };
    let y = unsafe{ std::slice::from_raw_parts(y_ptr, n) };
    let mut r = 0f64;

    for i in 0..n {
        r += x[i] * y[i];
    }
    return r;
}

// vector dot r = x ・ x
#[no_mangle]
pub extern fn DotSelf(x_ptr : *const f64, n : usize) -> f64 {
    let x = unsafe{ std::slice::from_raw_parts(x_ptr, n) };
    let mut r = 0f64;

    for i in 0..n {
        r += x[i] * x[i];
    }
    return r;
}

// vector copy y = x
#[no_mangle]
pub extern fn Copy(y_ptr : *mut f64, x_ptr : *const f64, n : usize) {
    let x = unsafe{ std::slice::from_raw_parts(x_ptr, n) };
    let y = unsafe{ std::slice::from_raw_parts_mut(y_ptr, n) };

    for i in 0..n {
        y[i] = x[i];
    }
}
