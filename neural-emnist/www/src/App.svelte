<div class="container">
    <div style="display: flex; flex-direction: column; width: 1282px;">
        <div style="margin: 8px 0;">
            <p style="float: left; margin: 0;">{ networkName ?? `Default network` }</p>
            <button on:click={upload} style="float: right;">Upload custom</button>
            <input type="file" bind:this={networkPicker} on:change={onNetworkPicked} style="display: none;" />
        </div>
    
        <DrawCanvas width={1280} height={720} characterSide={28} {debug} {editable} bind:getLines={getLines} bind:drawImage={drawImage} bind:clear={clearCanvas} />

        <button on:click={uploadImage} style="margin-top: 8px;">Upload image</button>
        <input type="file" accept=".jpg,.jpeg,.png" bind:this={imagePicker} on:change={onImagePicked} style="display: none;" />
    
        <p style="width: 100%; margin: 16px 0; text-align: center;">
            { #if result != null }
                Result: { result.map((v) => typeof v == 'number' ? v > medianKerning() * kerningMultiplier ? ' ' : '' : v).join('') }
            { :else }
                Press detect to recognize the text
            { /if }
        </p>
    
        <button on:click={detect}>Detect</button>
        <button on:click={clear} style="margin-top: 4px; margin-bottom: 16px;">Clear</button>
    
        { #if debug && result != null }
            <table>
                <thead>
                    <tr>
                        { #each result.filter((x) => typeof x == 'string') as character }
                            <th>{ character }</th>
                        { /each }
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        { #each result.filter((x) => typeof x == 'string') as _, i }
                            <td>{ Math.round(probabilities[i] * 100) / 100 }</td>
                        { /each }
                    </tr>
                </tbody>
            </table>
        { /if }

        <div bind:this={debugContainer}></div>
    </div>

    <div style="display: flex; justify-content: space-between; align-items: center; position: absolute; left: 0; right: 0; bottom: 0; padding: 8px; gap: 24px;">
        <div></div>

        <div style="display: flex; gap: 8px;">
            Kerning multiplier: { kerningMultiplier }
            <input type="range" bind:value={kerningMultiplier} min={1} max={5} step={0.1}>
        </div>

        <div style="display: flex; gap: 8px;">
            <input type="checkbox" id="debug" bind:checked={debug}>
            <label for="debug">Debug</label>
        </div>
    </div>
</div>

<script lang="ts">
    import { onMount } from 'svelte';
    import * as IJS from 'image-js';
    import wasmInit, * as wasm from 'wasm';

    import DrawCanvas from './DrawCanvas.svelte';

    import defaultNetworkPath from '../assets/default.nn64';

    let getLines: () => (number | number[][])[][];
    let drawImage: (image: HTMLImageElement) => void;
    let clearCanvas: () => void;

    let networkName: string | undefined;
    let network: wasm.Network;
    let networkPicker: HTMLInputElement;
    let networkReader = new FileReader();

    let imagePicker: HTMLInputElement;

    let kerningMultiplier = 2.5;

    let result: (string | number)[];
    let probabilities: number[];

    let debugContainer: HTMLDivElement;
    let debug = false;
    let editable = true;

    onMount(async () => {
        networkReader.addEventListener('load', onNetworkLoad);

        await wasmInit();

        fetch(defaultNetworkPath)
            .then((response) => response.arrayBuffer())
            .then((buffer) => network = new wasm.Network(new Uint8Array(buffer)))
            .catch((error) => console.error(`Error while loading default network: ${error}`));
    });

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

    function uploadImage() {
        imagePicker.click();
    }

    async function onImagePicked() {
        let file = imagePicker.files[0];

        if (file.type != 'image/jpeg' && file.type != 'image/png') {
            console.warn(`Unknown image type: ${file.type}`);
        }
        
        let image = await IJS.Image.load(URL.createObjectURL(file));
        image = image.grey({ algorithm: 'black' as IJS.GreyAlgorithm }).mask().invert();

        let imageElement = new Image();

        imageElement.onload = () => {
            console.log(imageElement);
            drawImage(imageElement);
            editable = false;
        }

        imageElement.src = image.toDataURL();
        console.log('pre', imageElement);
    }

    function debugLines(lines: (number | number[][])[][]) {
        for (let line of lines) {
            let lineContainer = document.createElement('div');

            for (let character of line) {
                if (typeof character == 'number') {
                    let spacing = document.createElement('p');

                    spacing.innerText = `${character}`;
                    spacing.style.opacity = '50%';
                    spacing.style.display = 'inline-block';

                    lineContainer.appendChild(spacing);
                } else {
                    let characterCanvas = document.createElement('canvas');

                    characterCanvas.width = 140;
                    characterCanvas.height = 140;

                    let characterContext = characterCanvas.getContext('2d');

                    for (let y = 0; y < character.length; y++) {
                        let row = character[y];

                        for (let x = 0; x < row.length; x++) {
                            if (row[x] > 0) {
                                let intesity = Math.round(255 * (1 - row[x]));
                                
                                characterContext.fillStyle = `rgb(${intesity}, ${intesity}, ${intesity})`;
                                characterContext.fillRect(x * 5, y * 5, 5, 5);
                            }
                        }
                    }

                    lineContainer.appendChild(characterCanvas);
                }
            }

            debugContainer.appendChild(lineContainer);
        }
    }

    function detect() {
        result = [];
        probabilities = [];

        let lines = getLines();

        if (debug) debugLines(lines);
        
        for (let line of lines) {
            for (let character of line) {
                if (typeof character != 'number') {
                    let data = character as number[][];
                    let transposedData = data.map((_, i) => data.map((r) => r[i]));
                    
                    let output = Array.from(network.feed_forward(new Float64Array(transposedData.flat())).entries());
                    let characterOutput = output.sort(([,p1], [,p2]) => p2 - p1)[0];

                    result.push(String.fromCharCode(characterOutput[0] + 96));
                    probabilities.push(characterOutput[1]);
                } else {
                    result.push(character);
                }
            }

            result.push(Infinity);
        }
    }

    function medianKerning(): number {
        let spaces = result.filter((character) => typeof character == 'number').sort() as number[];

        let half = Math.floor(spaces.length / 2);

        if (spaces.length % 2 == 0) {
            return (spaces[half - 1] + spaces[half]) / 2;
        } else {
            return spaces[half];
        }
    }

    function clear() {
        clearCanvas();
        editable = true;
        result = null;
        debugContainer.innerHTML = '';
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