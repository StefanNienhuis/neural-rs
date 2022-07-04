<div style="display: flex; flex-direction: column; width: 282px;">
    <div style="margin: 8px 0;">
        <p bind:this={networkLabel} style="float: left; margin: 0;">Default network</p>
        <button on:click={upload} style="float: right;">Upload</button>
        <input type="file" bind:this={networkPicker} on:change={onNetworkPicked} style="display: none;" />
    </div>

    <canvas bind:this={canvas} width="280" height="280" style="border: 1px solid lightgray; cursor: crosshair;"></canvas>

    <p bind:this={resultLabel} style="width: 100%; text-align: center;">Press detect to recognize the number</p>

    <button on:click={detect} style="margin-top: 4px;">Detect</button>
    <button on:click={clear} style="margin-top: 4px;">Clear</button>
</div>

<script lang="ts">
    import { onMount } from 'svelte';
    import { wasm } from './main';

    let canvas: HTMLCanvasElement;
    let context: CanvasRenderingContext2D;

    let networkLabel: HTMLParagraphElement;
    let network: wasm.Network;
    let networkPicker: HTMLInputElement;
    let networkReader = new FileReader();

    let resultLabel: HTMLParagraphElement;
    
    let pixels = new Array(28 * 28).fill(0);

    let mouseDown = false;

    onMount(() => {
        context = canvas.getContext('2d');

        canvas.addEventListener('mousedown', (event) => {
            mouseDown = true;
            drawMouse(event.clientX, event.clientY);
        });

        canvas.addEventListener('mousemove', (event) => drawMouse(event.clientX, event.clientY));
        canvas.addEventListener('mouseup', () => mouseDown = false);
        canvas.addEventListener('mouseleave', () => mouseDown = false);

        networkReader.addEventListener('load', onNetworkLoad);
    });

    // Canvas

    let lastX: number, lastY: number;

    function drawMouse(mouseX: number, mouseY: number) {
        if (!mouseDown) return;

        let canvasRect = canvas.getBoundingClientRect();

        let clientX = mouseX - canvasRect.left;
        let clientY = mouseY - canvasRect.top;

        let x = clamp(Math.floor(clientX / 10), 0, 27);
        let y = clamp(Math.floor(clientY / 10), 0, 27);

        if (lastX == x && lastY == y) return;

        lastX = x;
        lastY = y;

        pixels[y * 28 + x] = 1;

        if (x > 0) pixels[y * 28 + x - 1] += 0.2;
        if (x < 27) pixels[y * 28 + x + 1] += 0.2;
        if (y > 0) pixels[(y - 1) * 28 + x] += 0.2;
        if (y > 0) pixels[(y + 1) * 28 + x] += 0.2;

        draw(x, y);
    }

    function draw(x: number, y: number, edge: boolean = false) {
        let intensity = pixels[y * 28 + x];
        let color = Math.round(255 * (1 - intensity));

        context.fillStyle = `rgb(${color}, ${color}, ${color})`;

        context.fillRect(x * 10, y * 10, 10, 10);

        if (!edge) {
            draw(x, y - 1, true);
            draw(x, y + 1, true);
            draw(x - 1, y, true);
            draw(x + 1, y, true);
        }
    }

    function clamp(x: number, min: number, max: number) {
        return Math.min(Math.max(x, min), max);
    }

    // Neural network

    function upload() {
        networkPicker.click();
    }

    function onNetworkPicked() {
        let file = networkPicker.files[0];

        networkLabel.innerText = file.name;
        networkReader.readAsArrayBuffer(file);
    }

    function onNetworkLoad(event: ProgressEvent<FileReader>) {
        let result = event.target.result as ArrayBuffer;
        let data = new Uint8Array(result);

        network = new wasm.Network(data);
    }

    function detect() {
        let results = network.feed_forward(new Float64Array(pixels).map((x) => Math.min(x, 1)));
        let propability = Math.max(...results);
        let result = results.indexOf(propability);

        resultLabel.innerText = `Result: ${result} (${Math.round(propability * 1000) / 10}%)`
    }

    function clear() {
        pixels = pixels.fill(0);
        context.clearRect(0, 0, 280, 280);
    }
</script>