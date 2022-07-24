<div class="container">
    <div style="display: flex; flex-direction: column; width: 282px;">
        <div style="margin: 8px 0;">
            <p style="float: left; margin: 0;">{ networkName ?? 'Default network (digits)' }</p>
            <button on:click={upload} style="float: right;">Upload custom</button>
            <input type="file" bind:this={networkPicker} on:change={onNetworkPicked} style="display: none;" />
        </div>
    
        <DrawCanvas width={280} height={280} scaledSize={28} {editable} bind:getPixels={getPixels} bind:clear={clearCanvas} />

        <div style="width: 100%; margin-top: 16px; text-align: center;">
            <input type="radio" bind:group={type} id="digits" value="digits">
            <label for="digits">Digits</label>

            <input type="radio" bind:group={type} id="letters" value="letters">
            <label for="letters">Letters</label>
        </div>
    
        <p style="width: 100%; margin: 16px 0; text-align: center;">
            { #if results != null }
                { #if results[0][1] > 0.3 }
                    Result: { type === 'digits' ? results[0][0] : String.fromCharCode(results[0][0] + 96) } ({Math.round(results[0][1] * 1000) / 10}%)
                { :else }
                    Failed to recognize a number
                { /if }
            { :else }
                Press detect to recognize the { type === 'digits' ? 'digit' : 'letter' }
            { /if }
        </p>
    
        <button on:click={detect} disabled={network == null || !editable}>Detect</button>
        <button on:click={clear} style="margin-top: 4px; margin-bottom: 16px;">Clear</button>
    
        { #if results != null }
            <table>
                <thead>
                    <tr>
                        { #each results as [i] }
                            <th>{ type === 'digits' ? i : String.fromCharCode(i + 96) }</th>
                        { /each }
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        { #each results as [, p] }
                            <td>{Math.round(p * 1000) / 10}%</td>
                        { /each }
                    </tr>
                </tbody>
            </table>
        { /if }
    </div>
</div>

<script lang="ts">
    import { onMount } from 'svelte';
    import wasmInit, * as wasm from 'wasm';

    import DrawCanvas from './DrawCanvas.svelte';
    import defaultNetwork from '../assets/default.nnet';

    let getPixels: () => number[];
    let clearCanvas: () => void;

    let networkName: string | undefined;
    let network: wasm.Network;
    let networkPicker: HTMLInputElement;
    let networkReader = new FileReader();

    let type: 'digits' | 'letters' = 'digits';

    let results: [number, number][];
    let editable = true;

    onMount(async () => {
        networkReader.addEventListener('load', onNetworkLoad);

        await wasmInit();

        fetch(defaultNetwork)
            .then((response) => response.arrayBuffer())
            .then((buffer) => network = new wasm.Network(new Uint8Array(buffer)))
            .catch((error) => console.error(`Error while loading default network: ${error}`));
    });

    // Neural network

    function upload() {
        networkPicker.click();
    }

    function onNetworkPicked() {
        let file = networkPicker.files[0];

        networkName = file.name;
        networkReader.readAsArrayBuffer(file);
    }

    function onNetworkLoad(event: ProgressEvent<FileReader>) {
        let result = event.target.result as ArrayBuffer;
        let data = new Uint8Array(result);

        network = new wasm.Network(data);
    }

    function detect() {
        let pixels = getPixels();
        editable = false;
        
        results = Array.from(network.feed_forward(new Float64Array(pixels)).entries());

        results = results.sort(([, a], [, b]) => b - a).slice(0, 5);
    }

    function clear() {
        clearCanvas();
        editable = true;
        results = null;
    }
</script>

<style>
    .container {
        width: 100vw;
        height: 100vh;
        display: flex;
        justify-content: center;
        align-items: center;
    }

    table {
        table-layout: fixed;
        border-collapse: collapse;
        border-spacing: 0;
        width: 100%;
    }

    th, td {
        width: 10%;
        border: 1px solid black;
        text-align: center;
    }
</style>