// kira_lora_eq_ui.js
// Custom UI extension for Kira LoRA EQ node

import { app } from "../../scripts/app.js";

const EXTENSION_NAME = "kira.lora.eq.ui";
const NODE_CLASS = "KiraLora_EQ";

const FIXED_NODE_WIDTH = 460;
const FIXED_NODE_HEIGHT = 220;
const EQ_HEIGHT = 180;
const VALUE_STEP = 0.05;

const SKIN_IMAGE_URL = new URL("./kira_node_skin.png", import.meta.url).href;

let skinImage = null;
let skinLoaded = false;
let skinTried = false;

function ensureSkinLoaded() {
    if (skinLoaded || skinTried) return;
    skinTried = true;

    const img = new Image();
    img.onload = () => {
        skinImage = img;
        skinLoaded = true;
        if (app?.graph) app.graph.setDirtyCanvas(true, true);
    };
    img.onerror = () => {
        skinLoaded = false;
        skinImage = null;
    };
    img.src = SKIN_IMAGE_URL;
}

function quantizeValue(val, min, max) {
    let v = Math.round(val / VALUE_STEP) * VALUE_STEP;
    v = Math.max(min, Math.min(max, v));
    return parseFloat(v.toFixed(2));
}

function setupKiraEqNode(node) {
    if (node.__kiraEqPatched) return;
    node.__kiraEqPatched = true;

    if (!node.widgets || node.widgets.length === 0) {
        return;
    }

    const loraWidget = node.widgets.find((w) => w.name === "lora_name");
    const gainWidget = node.widgets.find((w) => w.name === "gain");
    const bandWidgets = node.widgets.filter(
        (w) => typeof w.name === "string" && w.name.startsWith("band_")
    );

    if (!gainWidget || bandWidgets.length === 0) {
        return;
    }

    if (loraWidget) loraWidget.label = "LoRA";
    gainWidget.label = "Master Gain";

    for (const w of bandWidgets) {
        const val = parseFloat(w.value);
        if (isNaN(val) || val <= 0) {
            w.value = 1.0;
        }
    }

    if (Array.isArray(node.inputs)) {
        node.inputs = node.inputs.filter(
            (inp) => inp?.name === "model" || inp?.name === "clip"
        );
    }

    node.size = [FIXED_NODE_WIDTH, FIXED_NODE_HEIGHT];
    node.resizable = false;

    node.computeSize = function () {
        const titleH = LiteGraph.NODE_TITLE_HEIGHT || 24;

        if (this.flags & LiteGraph.NODE_COLLAPSED || this.collapsed) {
            this.size[0] = FIXED_NODE_WIDTH;
            this.size[1] = titleH;
        } else {
            this.size[0] = FIXED_NODE_WIDTH;
            this.size[1] = FIXED_NODE_HEIGHT;
        }
        return this.size;
    };

    ensureSkinLoaded();

    const origOnDrawBackground = node.onDrawBackground
        ? node.onDrawBackground.bind(node)
        : null;

    node.onDrawBackground = function (ctx, canvas) {
        if (origOnDrawBackground) origOnDrawBackground(ctx, canvas);

        if (this.flags & LiteGraph.NODE_COLLAPSED || this.collapsed) return;

        if (skinLoaded && skinImage) {
            ctx.save();
            ctx.drawImage(skinImage, 0, 0, this.size[0], this.size[1]);
            ctx.restore();
        }
    };

    for (const w of bandWidgets) {
        w.computeSize = () => [0, 0];
        w.draw = () => {};
    }

    const eqWidget = {
        type: "kira_eq",
        name: "LoRA EQ",
        size: [FIXED_NODE_WIDTH, EQ_HEIGHT],
        bands: bandWidgets,
        gain: gainWidget,
        activeBand: null,
        last_y: 0,
        _dragging: false,
        _dragBandIndex: null,

        draw(ctx, node, width, y) {
            if (node.flags & LiteGraph.NODE_COLLAPSED || node.collapsed) return;

            const w = FIXED_NODE_WIDTH;
            const h = EQ_HEIGHT;
            this.size[0] = w;
            this.size[1] = h;
            this.last_y = y;

            const bands = this.bands || [];
            const n = bands.length;
            if (!n) return;

            const marginX = 10;
            const marginTop = 16;
            const marginBottom = 14;

            const barAreaWidth = w - marginX * 2;
            const barWidth = barAreaWidth / n;
            const barMaxHeight = h - marginTop - marginBottom;

            ctx.save();

            ctx.strokeStyle = "rgba(255,255,255,0.25)";
            ctx.strokeRect(0.5, y + 0.5, w - 1, h - 1);

            ctx.strokeStyle = "rgba(255,255,255,0.25)";
            ctx.beginPath();
            for (let i = 0; i <= 2; i++) {
                const gy = y + marginTop + (barMaxHeight * i) / 2;
                ctx.moveTo(marginX, gy + 0.5);
                ctx.lineTo(w - marginX, gy + 0.5);
            }
            ctx.stroke();

            for (let i = 0; i < n; i++) {
                const wBand = bands[i];
                if (!wBand) continue;

                const opts = wBand.options || {};
                const min = typeof opts.min === "number" ? opts.min : 0.0;
                const max = typeof opts.max === "number" ? opts.max : 1.5;

                let v = typeof wBand.value === "number" ? wBand.value : 1.0;
                v = quantizeValue(v, min, max);
                wBand.value = v;

                let norm = (v - min) / (max - min || 1);
                norm = Math.max(0, Math.min(1, norm));

                const x = marginX + i * barWidth;
                const barH = barMaxHeight * norm;
                const barY = y + marginTop + (barMaxHeight - barH);

                ctx.fillStyle = this.activeBand === i ? "#7cf" : "#aaf";
                ctx.fillRect(x + 3, barY, barWidth - 6, barH);

                ctx.fillStyle = "#ffffff";
                ctx.font = "9px sans-serif";
                ctx.textAlign = "center";
                ctx.textBaseline = "top";
                ctx.fillText(String(i + 1), x + barWidth / 2, y + 2);

                const displayVal = quantizeValue(v, min, max);
                ctx.fillStyle = "#ffffff";
                ctx.font = "9px sans-serif";
                ctx.textAlign = "center";
                ctx.textBaseline = "bottom";
                ctx.fillText(displayVal.toFixed(2), x + barWidth / 2, y + h - 2);
            }

            ctx.restore();
        },

        computeSize() {
            return [FIXED_NODE_WIDTH, EQ_HEIGHT];
        },

        async serializeValue() {
            return undefined;
        },

        mouse(event, nodePos, node) {
            if (node.flags & LiteGraph.NODE_COLLAPSED || node.collapsed) return false;

            const bands = this.bands || [];
            const n = bands.length;
            if (!n) return false;

            const type = event.type;
            const width = FIXED_NODE_WIDTH;
            const height = EQ_HEIGHT;

            const marginX = 10;
            const marginTop = 16;
            const marginBottom = 14;

            const barAreaWidth = width - marginX * 2;
            const barWidth = barAreaWidth / n;
            const barMaxHeight = height - marginTop - marginBottom;

            const localX = nodePos[0];
            const localY = nodePos[1] - this.last_y;

            const insideVertically = localY >= 0 && localY <= height;
            const insideHorizontally = localX >= 0 && localX <= width;

            const clamp01 = (v) => Math.max(0, Math.min(1, v));

            const getBandIndexAndValue = () => {
                if (!insideVertically || !insideHorizontally) return null;

                let bandIndex = Math.floor((localX - marginX) / barWidth);
                bandIndex = Math.max(0, Math.min(n - 1, bandIndex));

                let norm = 1.0 - (localY - marginTop) / barMaxHeight;
                norm = clamp01(norm);

                const wBand = bands[bandIndex];
                if (!wBand) return null;

                const opts = wBand.options || {};
                const min = typeof opts.min === "number" ? opts.min : 0.0;
                const max = typeof opts.max === "number" ? opts.max : 1.5;

                const rawVal = min + norm * (max - min);
                const val = quantizeValue(rawVal, min, max);

                return { bandIndex, val };
            };

            if (type === "pointerdown" || type === "mousedown") {
                const info = getBandIndexAndValue();
                if (!info) return false;

                const { bandIndex, val } = info;
                const wBand = bands[bandIndex];
                if (!wBand) return false;

                wBand.value = val;
                if (wBand.callback) wBand.callback(wBand.value, node);

                if (node.graph) {
                    node.graph._version++;
                }

                this.activeBand = bandIndex;
                this._dragging = true;
                this._dragBandIndex = bandIndex;

                node.setDirtyCanvas(true, true);
                return true;
            }

            if ((type === "pointermove" || type === "mousemove") && this._dragging) {
                if (this._dragBandIndex == null) return false;
                if (!insideVertically) return false;

                const bandIndex = this._dragBandIndex;
                const wBand = bands[bandIndex];
                if (!wBand) return false;

                const opts = wBand.options || {};
                const min = typeof opts.min === "number" ? opts.min : 0.0;
                const max = typeof opts.max === "number" ? opts.max : 1.5;

                let norm = 1.0 - (localY - marginTop) / barMaxHeight;
                norm = clamp01(norm);

                const rawVal = min + norm * (max - min);
                const val = quantizeValue(rawVal, min, max);

                wBand.value = val;
                if (wBand.callback) wBand.callback(wBand.value, node);

                if (node.graph) {
                    node.graph._version++;
                }

                this.activeBand = bandIndex;
                node.setDirtyCanvas(true, true);
                return true;
            }

            if (type === "pointerup" || type === "mouseup") {
                if (this._dragging) {
                    this._dragging = false;
                    this._dragBandIndex = null;
                    return true;
                }
                return false;
            }

            return false;
        },
    };

    node.addCustomWidget(eqWidget);

    const idxEq = node.widgets.indexOf(eqWidget);
    const idxGain = node.widgets.indexOf(gainWidget);
    if (idxEq > -1 && idxGain > -1 && idxEq !== idxGain + 1) {
        node.widgets.splice(idxEq, 1);
        node.widgets.splice(idxGain + 1, 0, eqWidget);
    }

    node.__kiraEqWidget = eqWidget;
    node.__kiraEqBands = bandWidgets;

    node.size = [FIXED_NODE_WIDTH, FIXED_NODE_HEIGHT];
    node.setDirtyCanvas(true, true);
}

app.registerExtension({
    name: EXTENSION_NAME,
    async nodeCreated(node) {
        if (node?.comfyClass === NODE_CLASS) {
            setupKiraEqNode(node);
        }
    },
});
