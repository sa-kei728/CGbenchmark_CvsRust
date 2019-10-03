#![crate_type = "bin"]

fn array_print<T>(v: &T) where
T : std::ops::Index<usize, Output=f64> + ?Sized //?SizedがないとSliceがNG
{
    println!("{}", v[2]);   //v[3]にすると実行時Abortする…vecのサイズが不定なため?
}

fn main(){
    //let ar   = [1f64; 3]; //固定長のスタック配列はコンパイルが通らない…原因確認が要るかも
    let vec  = vec![2f64; 3];
    let vec2 = vec![3f64; 3];
    let sli  = &vec2[..];

    //array_print(&ar);
    array_print(&vec);
    array_print(sli);
}