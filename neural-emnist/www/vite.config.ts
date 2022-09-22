import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';

export default defineConfig({
    plugins: [svelte()],
    server: {
        fs: {
            allow: ['.', '../pkg']
        }
    },
    assetsInclude: ['**/*.nn64', '**/*.nn32']
});
