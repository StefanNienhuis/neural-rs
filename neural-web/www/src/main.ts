import App from './App.svelte';
import init, * as wasm from 'wasm';

const app = new App({
    target: document.getElementById('app')
});

await init();

export default app;
export { wasm };