<canvas bind:this={canvas} width={width} height={height} style={`border: 1px solid lightgray; cursor: ${editable ? 'crosshair' : 'not-allowed'};`}></canvas>

<script lang="ts">
    import { onMount } from "svelte";

    export let width: number;
    export let height: number;
    export let characterSide: number;
    export let debug: boolean;
    export let editable: boolean;

    let canvas: HTMLCanvasElement;
    let context: CanvasRenderingContext2D;
    let mouseDown = false;

    let minX = 0;
    let minY = 0;
    let maxX = 0;
    let maxY = 0;

    let lastX, lastY;
    
    onMount(() => {
        context = canvas.getContext('2d');
        context.lineWidth = 30;

        canvas.addEventListener('mousedown', (event) => drawStart(event.clientX, event.clientY));
        canvas.addEventListener('mousemove', (event) => draw(event.clientX, event.clientY));
        canvas.addEventListener('mouseup', drawEnd);
        canvas.addEventListener('mouseleave', drawEnd);

        canvas.addEventListener('touchstart', (event) => { event.preventDefault(); drawStart(event.touches[0].clientX, event.touches[0].clientY); });
        canvas.addEventListener('touchmove', (event) => { event.preventDefault(); draw(event.touches[0].clientX, event.touches[0].clientY); });
        canvas.addEventListener('touchend', drawEnd);
        canvas.addEventListener('touchcancel', drawEnd);

        minX = width;
        minY = height;
    });

    function drawStart(x: number, y: number) {
        mouseDown = true;
        draw(x, y);
    }

    function drawEnd() {
        mouseDown = false;
        lastX = null;
        lastY = null;
    }

    function draw(mouseX: number, mouseY: number) {
        if (!mouseDown || !editable) return;

        let canvasRect = canvas.getBoundingClientRect();

        let x = mouseX - canvasRect.left;
        let y = mouseY - canvasRect.top;

        if (x - 15 <= minX) minX = x - 15;
        if (x + 15 >= maxX) maxX = x + 15;
        if (y - 15 <= minY) minY = y - 15;
        if (y + 15 >= maxY) maxY = y + 15;

        if (lastX != null && lastY != null) {
            context.beginPath();

            context.moveTo(lastX, lastY);
            context.lineTo(x, y);
            context.stroke();

            context.closePath();
        }

        context.beginPath();

        context.arc(x, y, 15, 0, 2 * Math.PI);
        context.fill();

        context.closePath();

        lastX = x;
        lastY = y;
    }

    export const drawImage = (image: HTMLImageElement) => {
        let scaleFactor = Math.min(image.width / width, image.height / height);
        canvas.width = image.width / scaleFactor;
        canvas.height = image.height / scaleFactor;

        minX = 0;
        minY = 0;
        maxX = canvas.width;
        maxY = canvas.height;
        
        context.drawImage(image, 0, 0, canvas.width, canvas.height);
        context.lineWidth = 30;
    }

    export const getLines = () => {
        let data = context.getImageData(minX, minY, maxX - minX, maxY - minY);

        // Convert image data to pixel array
        // Pixel array has shape pixels[y][x]
        let pixels: number[][] = Array(data.height).fill(null).map(() => []);
        
        for (let i = 0; i < data.data.length / 4; i++) {
            let grayscale = (data.data[i * 4] + data.data[i * 4 + 1] + data.data[i * 4 + 2]) / 3 / 255;
            let alpha = data.data[i * 4 + 3] / 255;
            
            pixels[Math.floor(i / data.width)][i % data.width] = Math.round((1 - grayscale) * alpha);
        }

        // Extract lines from pixels
        let lineBounds: [number, number][] = [];

        let lineStart: number | undefined;

        for (let y = 0; y < pixels.length; y++) {
            if (pixels[y].some(intensity => intensity > 0)) {
                if (lineStart == null) {
                    lineStart = y;
                }
            } else if (lineStart != null) {
                lineBounds.push([lineStart, y]);
                lineStart = null;
            }
        }

        // If last line is not completed, take until last row
        if (lineStart != null) {
            lineBounds.push([lineStart, pixels.length]);
        }

        // List of lines
        // Each line is a list of characters or spacing between characters
        // Each character is a 2d array of pixels
        let lines: (number[][] | number)[][] = [];

        for (let [lineStart, lineEnd] of lineBounds) {
            if (debug) {
                context.lineWidth = 2;
                context.strokeStyle = 'red';

                context.beginPath();
                context.moveTo(0, lineStart + minY);
                context.lineTo(canvas.width, lineStart + minY);

                context.moveTo(0, lineEnd + minY);
                context.lineTo(canvas.width, lineEnd + minY);
                context.closePath();

                context.stroke();
            }

            let line = pixels.slice(lineStart, lineEnd);
            let lineWidth = line[0].length;

            // Extract characters from line
            let characterBounds: [number, number][] = [];

            let characterStart: number | undefined;
            let characterEnd: number | undefined;

            for (let x = 0; x < lineWidth; x++) {
                if (line.map(row => row[x]).some(intensity => intensity > 0)) {
                    if (characterStart == null) {
                        characterStart = x;
                    }
                } else if (characterStart != null) {
                    // Allow a 5% margin for missing pixels
                    if (characterEnd != null && (x - characterEnd) / (characterEnd - characterStart) > 0.05) {
                        characterBounds.push([characterStart, characterEnd]);
                        characterStart = null;  
                        characterEnd = null;
                    } else if (characterEnd == null) {
                        characterEnd = x;
                    }
                }
            }

            // If last character is not completed, take until last column
            if (characterStart != null) {
                characterBounds.push([characterStart, lineWidth]);
            }

            // Array of 2d character arrays and the spacing between those characters
            let characters: (number[][] | number)[] = [];

            for (let [i, [start, end]] of characterBounds.entries()) {
                if (debug) {
                    context.strokeStyle = 'blue';

                    context.beginPath();
                    context.moveTo(start + minX, lineStart + minY);
                    context.lineTo(start + minX, lineEnd + minY);

                    context.moveTo(end + minX, lineStart + minY);
                    context.lineTo(end + minX, lineEnd + minY);
                    context.closePath();

                    context.stroke();
                }

                let data = line.map((row) => row.filter((_, x) => x >= start && x < end));

                let startY = 0, endY = data.length - 1;

                // Remove top and bottom spacing from other characters in line
                while (data[0].every((x) => x == 0)) { data.shift(); startY++; }
                while (data[data.length - 1].every((x) => x == 0)) { data.pop(); endY--; }

                if (debug) {
                    context.strokeStyle = 'green';

                    context.beginPath();
                    context.moveTo(start + minX, startY + lineStart + minY);
                    context.lineTo(end + minX, startY + lineStart + minY);

                    context.moveTo(start + minX, endY + lineStart + minY);
                    context.lineTo(end + minX, endY + lineStart + minY);
                    context.closePath();

                    context.stroke();
                }
                                
                let dataWidth = data[0].length;
                let dataHeight = data.length;

                let side = Math.max(dataWidth, dataHeight);
                side *= 1.2; // Add 10% padding on each side
                side = Math.round(side);

                if (dataWidth > side || dataHeight > side) {
                    console.error(`Invalid side lengths: ${dataWidth}, ${dataHeight} should be less than square ${side}`);
                    return;
                }

                let xPadding = side - dataWidth;
                let yPaddding = side - dataHeight;

                // Apply x padding
                data = data.map((row) => [
                    ...Array(Math.ceil(xPadding / 2)).fill(0),
                    ...row,
                    ...Array(Math.floor(xPadding / 2)).fill(0)
                ]);

                // Apply y padding
                data = [
                    ...Array(Math.ceil(yPaddding / 2)).fill(Array(side).fill(0)),
                    ...data,
                    ...Array(Math.floor(yPaddding / 2)).fill(Array(side).fill(0))
                ];

                let scaledData = scaleCharacter(data, side, characterSide);
                
                characters.push(scaledData);

                let nextCharacter = characterBounds[i + 1];

                if (nextCharacter != null) {
                    characters.push(nextCharacter[0] - end);
                }
            }

            lines.push(characters);
        }

        return lines;
    }

    function scaleCharacter(data: number[][], side: number, scaledSide: number) {
        let scaledData: number[][] = Array(scaledSide).fill(null).map(() => []);

        for (let scaledY = 0; scaledY < scaledSide; scaledY++) {
            for (let scaledX = 0; scaledX < scaledSide; scaledX++) {
                let x = scaledX * ((side - 1) / (scaledSide - 1));
                let y = scaledY * ((side - 1) / (scaledSide - 1));

                let fx = Math.floor(x);
                let cx = Math.ceil(x);
                let fy = Math.floor(y);
                let cy = Math.ceil(y);

                let intensity, a, b;
                
                if (fx == cx) {
                    a = data[fy]?.[fx] ?? 0;
                    b = data[cy]?.[fx] ?? 0;
                } else {
                    a = (x - fx) * (data[fy]?.[cx] ?? 0) + (cx - x) * (data[fy]?.[fx] ?? 0);
                    b = (x - fx) * (data[cy]?.[cx] ?? 0) + (cx - x) * (data[cy]?.[fx] ?? 0);
                }

                if (fy == cy) {
                    intensity = a;
                } else {
                    intensity = (y - fy) * b + (cy - y) * a;
                }

                if (Number.isNaN(intensity) || intensity == undefined) {
                    console.error(`Bilinear interpolation error: Intesity is NaN or undefined.`);
                    debugger;
                }

                scaledData[scaledY][scaledX] = intensity;
            }
        }

        return scaledData;
    }

    export const clear = () => {
        canvas.width = width;
        canvas.height = height;

        context.clearRect(0, 0, canvas.width, canvas.height);
        context.fillStyle = 'black';
        context.lineWidth = 30;

        minX = width;
        minY = height;
        maxX = 0;
        maxY = 0;
    };
</script>