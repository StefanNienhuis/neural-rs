mod utils;

use wasm_bindgen::prelude::*;

#[wasm_bindgen(start)]
pub fn main() {
    utils::set_panic_hook();
}

#[wasm_bindgen]
extern {
    fn alert(s: &str);
}

#[wasm_bindgen]
pub fn greet() {
    alert("Hello, World!");
}