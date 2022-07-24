<canvas bind:this={canvas} {width} {height} style={`border: 1px solid lightgray; cursor: ${editable ? 'crosshair' : 'not-allowed'};`}></canvas>

<script lang="ts">
    import { onMount } from "svelte";

    export let width: number;
    export let height: number;
    export let scaledSize: number;
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

        if (x <= minX) minX = x;
        if (x >= maxX) maxX = x;
        if (y <= minY) minY = y;
        if (y >= maxY) maxY = y;

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

    export const getPixels = () => {
        let side = Math.max(maxX - minX, maxY - minY) + 75;
        let centerX = Math.round((maxX + minX) / 2);
        let centerY = Math.round((maxY + minY) / 2);
        let data = context.getImageData(centerX - (side / 2), centerY - (side / 2), side, side);

        let pixels = [];
        
        for (let i = 0; i < data.data.length / 4; i++) {
            let intensity = data.data[(i * 4) + 3];
            
            pixels[i] = intensity / 255;
        }

        let scaledPixels = [];

        context.clearRect(0, 0, canvas.width, canvas.height);

        // Bilinear interpolation
        for (let row = 0; row < scaledSize; row++) {
            for (let column = 0; column < scaledSize; column++) {
                let x = column * ((side - 1) / (scaledSize - 1));
                let y = row * ((side - 1) / (scaledSize - 1));

                let fx = Math.floor(x);
                let cx = Math.ceil(x);
                let fy = Math.floor(y);
                let cy = Math.ceil(y);

                let intensity, a, b;

                if (fx == cx) {
                    a = pixels[fy * side + fx] || 0;
                    b = pixels[cy * side + fx] || 0;
                } else {
                    a = (x - fx) * (pixels[fy * side + cx] || 0) + (cx - x) * (pixels[fy * side + fx] || 0);
                    b = (x - fx) * (pixels[cy * side + cx] || 0) + (cx - x) * (pixels[cy * side + fx] || 0);
                }

                if (fy == cy) {
                    intensity = a;
                } else {
                    intensity = (y - fy) * b + (cy - y) * a;
                }

                if (Number.isNaN(intensity) || intensity == undefined) debugger;

                scaledPixels[column * scaledSize + row] = intensity;

                context.fillStyle = `rgba(0, 0, 0, ${intensity})`;
                context.fillRect(column * 10, row * 10, 10, 10);
            }
        }

        return scaledPixels;
    };

    export const clear = () => {
        context.clearRect(0, 0, canvas.width, canvas.height);
        context.fillStyle = 'black';

        minX = width;
        minY = height;
        maxX = 0;
        maxY = 0;
    };
</script>