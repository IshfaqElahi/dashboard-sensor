// =========================================================================
// CORE AVIONICS CALCULATION MODEL — GAUSSIAN DISPERSION INVERSION MATRIX
// =========================================================================

const PG = {
    A: { ay: 0.22, by: 0.9, az: 0.20, bz: 0.9 },
    B: { ay: 0.16, by: 0.9, az: 0.12, bz: 0.9 },
    C: { ay: 0.11, by: 0.9, az: 0.08, bz: 0.9 },
    D: { ay: 0.08, by: 0.9, az: 0.06, bz: 0.9 },
    E: { ay: 0.06, by: 0.9, az: 0.03, bz: 0.9 },
    F: { ay: 0.04, by: 0.9, az: 0.016, bz: 0.9 },
};

const WMO = {
    0: 'Clear', 1: 'Mainly Clear', 2: 'Scattered', 3: 'Overcast', 45: 'Rad Fog',
    51: 'Light Drizzle', 61: 'Light Rain', 63: 'Rain Vol', 65: 'Heavy Rain', 80: 'Showers', 95: 'T-Storm Active'
};
const DIR_NAMES = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'];
const PPM_TO_UGM3_H2 = 82.4;

let SRC = { lat: 23.0942, lon: 90.0157, alt: 30 };
let W = { U: 2.6, WD: 315, temp: 28, hum: 28, code: 0 };
let AQ = {};
let windMode = 'live';
let plumeCache = null;
let droneMarker = null;
let pickingMode = false;

delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({ iconUrl: '', iconRetinaUrl: '', shadowUrl: '' });

const map = L.map('map', {
    center: [SRC.lat, SRC.lon], zoom: 14,
    zoomControl: false, attributionControl: false
});

L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
    { subdomains: 'abcd', maxZoom: 19 }).addTo(map);
L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    { maxZoom: 19, opacity: 0.15 }).addTo(map);
L.control.zoom({ position: 'bottomright' }).addTo(map);
L.control.attribution({ position: 'bottomright', prefix: 'CartoDB/Esri/Open-Meteo' }).addTo(map);

function makeDroneIcon() {
    return L.divIcon({
        html: '<div class="drone-marker-pulsing" style="width:16px;height:16px;border-radius:50%;background:var(--drone);border:2px solid #fff;box-shadow:0 0 18px var(--drone);"></div>',
        iconSize: [16, 16], iconAnchor: [8, 8], className: ''
    });
}

function placeDroneMarker() {
    if (droneMarker) droneMarker.remove();
    droneMarker = L.marker([SRC.lat, SRC.lon], { icon: makeDroneIcon(), zIndexOffset: 1000 })
        .bindTooltip('ISOTOPER B15 INTERCEPT HUB (' + SRC.alt + 'm AGL)', { direction: 'right' })
        .addTo(map);
}
placeDroneMarker();

map.on('click', function (e) {
    if (!pickingMode) return;
    SRC.lat = +e.latlng.lat.toFixed(6);
    SRC.lon = +e.latlng.lng.toFixed(6);
    document.getElementById('inLat').value = SRC.lat;
    document.getElementById('inLon').value = SRC.lon;
    finishPick();
    applyDronePosition(true);
});

function pickFromMap() {
    pickingMode = true;
    document.getElementById('pickMsg').style.display = 'block';
    document.getElementById('mapWrap').style.cursor = 'crosshair';
}
window.pickFromMap = pickFromMap;

function finishPick() {
    pickingMode = false;
    document.getElementById('pickMsg').style.display = 'none';
    document.getElementById('mapWrap').style.cursor = '';
}

function applyDronePosition(skipPan) {
    const lat = parseFloat(document.getElementById('inLat').value);
    const lon = parseFloat(document.getElementById('inLon').value);
    const alt = parseFloat(document.getElementById('inAlt').value) || 30;
    if (isNaN(lat) || isNaN(lon) || lat < -90 || lat > 90 || lon < -180 || lon > 180) {
        document.getElementById('inLat').classList.add('invalid');
        document.getElementById('inLon').classList.add('invalid');
        return;
    }
    document.getElementById('inLat').classList.remove('invalid');
    document.getElementById('inLon').classList.remove('invalid');
    SRC.lat = lat; SRC.lon = lon; SRC.alt = alt;
    document.getElementById('Heff').value = alt;
    document.getElementById('lH').textContent = alt;
    placeDroneMarker();
    updateDroneHUD();
    if (!skipPan) map.flyTo([lat, lon], map.getZoom(), { duration: 0.8 });
    recomputePlume();
}
window.applyDronePosition = applyDronePosition;

function resetDroneToDefault() {
    SRC = { lat: 23.0942, lon: 90.0157, alt: 30 };
    document.getElementById('inLat').value = SRC.lat;
    document.getElementById('inLon').value = SRC.lon;
    document.getElementById('inAlt').value = SRC.alt;
    document.getElementById('Heff').value = SRC.alt;
    document.getElementById('lH').textContent = SRC.alt;
    placeDroneMarker();
    updateDroneHUD();
    map.flyTo([SRC.lat, SRC.lon], 14, { duration: 0.8 });
    recomputePlume();
}
window.resetDroneToDefault = resetDroneToDefault;

function updateDroneHUD() {
    document.getElementById('dronePos').textContent =
        SRC.lat.toFixed(5) + '°N  ' + SRC.lon.toFixed(5) + '°E';
    document.getElementById('droneAlt').textContent = 'ALT STACK: ' + SRC.alt + 'm AGL';
    document.getElementById('srcLatDisp').textContent = SRC.lat.toFixed(6);
    document.getElementById('srcLonDisp').textContent = SRC.lon.toFixed(6);
    document.getElementById('srcAltDisp').textContent = SRC.alt + ' m';
}

function setWindMode(mode) {
    windMode = mode;
    document.getElementById('tabLive').classList.toggle('active', mode === 'live');
    document.getElementById('tabManual').classList.toggle('active', mode === 'manual');
    document.getElementById('manualWindPanel').style.display = mode === 'manual' ? 'block' : 'none';
    if (mode === 'manual') {
        W.U = +document.getElementById('manWS').value;
        W.WD = +document.getElementById('manWD').value;
        updateUI(); recomputePlume();
    }
}
window.setWindMode = setWindMode;

document.getElementById('manWS').addEventListener('input', function () {
    W.U = +this.value;
    document.getElementById('lWS').textContent = W.U.toFixed(1);
    if (windMode === 'manual') { updateUI(); recomputePlume(); }
});
document.getElementById('manWD').addEventListener('input', function () {
    W.WD = +this.value;
    document.getElementById('lWD').textContent = W.WD;
    if (windMode === 'manual') { updateUI(); recomputePlume(); }
});

function log(m) { document.getElementById('loaderLog').textContent = m; }

async function fetchWeatherDirect(url) {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Weather system telemetry breach: ${response.status}`);
    return await response.json();
}

function adcToPPM(adc, Ro, RL) {
    var v = adc * (3.3 / 4095.0);
    if (v < 0.001) return 0;
    var Vc = 5.0;
    var Rs = ((Vc - v) / v) * RL;
    var ratio = Rs / Ro;
    return Math.max(0, 987.99 * Math.pow(Math.max(ratio, 0.001), -2.162));
}

async function fetchSheetData() {
    const scriptURL = "https://script.google.com/macros/s/AKfycbyfaKegzXJITJhS6Po5G0ir-y6ikMA1xN4rc3Oo3CEObOC9700473MPbuTnazip3fCn/exec";
    document.getElementById('sheetStatus').textContent = 'Synchronizing sensor array registers...';

    try {
        const response = await fetch(scriptURL);
        const data = await response.json();

        if (data.error) throw new Error(data.error);
        if (!data.latest) throw new Error('Missing "latest" data object from API');

        const env = data.latest;
        const adc = parseFloat(env.h2_raw);
        const ws = parseFloat(env.ws || env.wind_speed || W.U);
        const wd = parseFloat(env.wd || env.wind_direction || W.WD);

        if (isNaN(adc) || adc <= 0) throw new Error('Invalid or missing h2_raw value');

        const Ro = parseFloat(document.getElementById('mq8Ro').value) || 10;
        const RL = parseFloat(document.getElementById('mq8RL').value) || 10;

        const volt = adc * (3.3 / 4095.0);
        const Rs = ((5.0 - volt) / volt) * RL;
        const ratio = Rs / Ro;
        const ppm = adcToPPM(adc, Ro, RL);
        const C_ug_m3 = ppm * PPM_TO_UGM3_H2;

        document.getElementById('mq8ADC').textContent = adc.toFixed(0);
        document.getElementById('mq8Volt').textContent = volt.toFixed(3) + ' V';
        document.getElementById('mq8RsRo').textContent = ratio.toFixed(3);
        document.getElementById('mq8PPM').textContent = ppm.toFixed(2) + ' ppm';
        document.getElementById('sheetStatus').textContent = `Bound: ${C_ug_m3.toFixed(2)} µg/m³ H₂`;

        if (!isNaN(ws) && !isNaN(wd) && windMode === 'live') {
            W.U = ws;
            W.WD = wd;
        }

        updateDynamicQ(C_ug_m3);
        updateUI();
        recomputePlume();

    } catch (err) {
        console.error("Apps Script Integration Fault:", err);
        document.getElementById('sheetStatus').textContent = 'Fault: ' + err.message;
    }
}
window.fetchSheetData = fetchSheetData;

function updateDynamicQ(C_ug_m3) {
    if (C_ug_m3 <= 0) return;

    const H = parseFloat(document.getElementById('Heff').value);
    const stab = document.getElementById('stability').value;
    const U = Math.max(W.U, 0.2);

    const x_measure = 1.0;
    const y_measure = 0.0;
    const z_measure = SRC.alt;

    const { sy, sz } = getSigmas(x_measure, stab);

    const expY = Math.exp(-(y_measure * y_measure) / (2 * sy * sy));
    const expZ1 = Math.exp(-((z_measure - H) * (z_measure - H)) / (2 * sz * sz));
    const expZ2 = Math.exp(-((z_measure + H) * (z_measure + H)) / (2 * sz * sz));
    const D = (1 / (2 * Math.PI * sy * sz * U)) * expY * (expZ1 + expZ2);

    const Q_ug_s = C_ug_m3 / D;
    const Q_kg_s = Q_ug_s * 1e-9;

    const qSlider = document.getElementById('Q');
    if (Q_kg_s > parseFloat(qSlider.max)) qSlider.max = (Q_kg_s * 1.5).toFixed(5);
    if (Q_kg_s < parseFloat(qSlider.min)) qSlider.min = (Q_kg_s * 0.5).toFixed(5);

    qSlider.value = Q_kg_s;
    document.getElementById('lQ').textContent = Q_kg_s.toFixed(6);
}

async function fetchLiveWeather() {
    if (windMode === 'manual') return;
    document.getElementById('loader').style.display = 'flex';
    document.getElementById('hudST').textContent = 'SCANNING...';
    document.getElementById('hudST').style.color = 'var(--yellow)';
    document.getElementById('btnRefresh').disabled = true;

    const fURL = 'https://api.open-meteo.com/v1/forecast?latitude=' + SRC.lat +
        '&longitude=' + SRC.lon +
        '&current=temperature_2m,relative_humidity_2m,weather_code,surface_pressure,' +
        'wind_speed_10m,wind_direction_10m,wind_gusts_10m&wind_speed_unit=ms&timezone=auto';
    const aqURL = 'https://air-quality-api.open-meteo.com/v1/air-quality?latitude=' + SRC.lat +
        '&longitude=' + SRC.lon + '&current=pm10,pm2_5,nitrogen_dioxide,carbon_monoxide,ozone,european_aqi';

    try {
        log('Intercepting meteorological frames for Dhaka (' + SRC.lat.toFixed(4) + ', ' + SRC.lon.toFixed(4) + ')…');

        const [fData, aqJson] = await Promise.all([
            fetchWeatherDirect(fURL).catch(e => { return { current: {} }; }),
            fetchWeatherDirect(aqURL).catch(e => { return { current: {} }; })
        ]);

        log('Injecting environmental telemetry parameters into plume engine…');
        const cur = fData.current || {};

        const newWindSpeed = typeof cur.wind_speed_10m === 'number' ? cur.wind_speed_10m : W.U;
        const newWindDir = typeof cur.wind_direction_10m === 'number' ? cur.wind_direction_10m : W.WD;

        if (newWindSpeed !== W.U || newWindDir !== W.WD) {
            W.U = newWindSpeed;
            W.WD = newWindDir;
        }

        W.temp = typeof cur.temperature_2m === 'number' ? cur.temperature_2m : W.temp;
        W.hum = typeof cur.relative_humidity_2m === 'number' ? cur.relative_humidity_2m : W.hum;
        W.code = typeof cur.weather_code === 'number' ? cur.weather_code : W.code;
        W.gust = typeof cur.wind_gusts_10m === 'number' ? cur.wind_gusts_10m : null;

        const aqc = aqJson.current || {};
        AQ.pm25 = aqc.pm2_5 ?? null;
        AQ.pm10 = aqc.pm10 ?? null;
        AQ.no2 = aqc.nitrogen_dioxide ?? null;
        AQ.aqi = aqc.european_aqi ?? null;

        updateUI();
        recomputePlume();

        const now = new Date().toLocaleTimeString('en-BD', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
        document.getElementById('sTime').textContent = now;
        document.getElementById('hudTime').textContent = 'Synchronized ' + now;
        document.getElementById('hudST').textContent = 'ONLINE';
        document.getElementById('hudST').style.color = 'var(--green)';

    } catch (err) {
        console.error('Sensor connection dropped:', err);
        log('⚠ Telemetry fault recovery loop active — executing safe defaults');
        document.getElementById('hudST').textContent = 'FALLBACK';
        document.getElementById('hudST').style.color = 'var(--orange)';
        updateUI();
        recomputePlume();
    }

    document.getElementById('loader').style.display = 'none';
    document.getElementById('btnRefresh').disabled = false;
}
window.fetchLiveWeather = fetchLiveWeather;

const dirN = d => DIR_NAMES[Math.round(d / 22.5) % 16];
const fmt = (v, u, dp = 1) => (v != null) ? v.toFixed(dp) + ' ' + u : 'N/A';
const aqiC = v => !v ? 'var(--text)' : v < 20 ? 'var(--green)' : v < 40 ? '#10b981' : v < 60 ? 'var(--yellow)' : v < 80 ? 'var(--orange)' : 'var(--red)';

function updateUI() {
    const U = W.U, WD = W.WD;
    const plumeToward = (WD + 180) % 360;
    document.getElementById('lcWind').textContent = U.toFixed(1) + ' m/s' + (W.gust ? ' (g:' + W.gust.toFixed(1) + ')' : '');
    document.getElementById('lcDir').textContent = WD + '° FROM ' + dirN(WD);
    document.getElementById('lcPlumeDir').textContent = plumeToward + '° TO ' + dirN(plumeToward);
    document.getElementById('lcTemp').textContent = fmt(W.temp, '°C');
    document.getElementById('lcHum').textContent = fmt(W.hum, '%', 0);
    document.getElementById('lcCode').textContent = WMO[W.code] || ('Code ' + W.code);
    document.getElementById('aqPm25').textContent = fmt(AQ.pm25, 'µg/m³');
    document.getElementById('aqPm10').textContent = fmt(AQ.pm10, 'µg/m³');
    document.getElementById('aqNo2').textContent = fmt(AQ.no2, 'µg/m³');
    const ae = document.getElementById('aqAqi');
    ae.textContent = AQ.aqi ?? 'N/A'; ae.style.color = aqiC(AQ.aqi);

    document.getElementById('wstrip').innerHTML =
        '<span>Wind: <span class="wv">' + U.toFixed(1) + ' m/s ' + dirN(WD) + '</span></span>' +
        '<span>➔ <span class="wv">' + plumeToward + '° ' + dirN(plumeToward) + '</span></span>' +
        '<span>Temp: <span class="wv">' + W.temp.toFixed(1) + '°C</span></span>' +
        '<span>RH: <span class="wv">' + W.hum + '%</span></span>' +
        '<span>Forecast: <span class="wv">' + (WMO[W.code] || W.code) + '</span></span>' +
        '<span>AQI: <span class="wv" style="color:' + aqiC(AQ.aqi) + '">' + (AQ.aqi || '—') + '</span></span>';

    document.getElementById('sWind').textContent = U.toFixed(2) + ' m/s';
    document.getElementById('sWindDir').textContent = WD + '° FROM ' + dirN(WD);
    drawCompass(WD);
    document.getElementById('hudWT').textContent = 'Vector Flow: ' + U.toFixed(1) + ' m/s FROM ' + dirN(WD) + ' (' + WD + '°)';
}

function drawCompass(wd) {
    const cv = document.getElementById('compass');
    const ctx = cv.getContext('2d');
    ctx.clearRect(0, 0, 54, 54);
    const cx = 27, cy = 27, r = 21;

    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';

    ctx.strokeStyle = isDark ? '#4B5563' : '#9CA3AF'; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(cx, cy, r, 0, 2 * Math.PI); ctx.stroke();

    for (let i = 0; i < 8; i++) {
        const a = i * Math.PI / 4;
        ctx.strokeStyle = isDark ? '#6B7280' : '#6B7280'; ctx.lineWidth = 1; ctx.beginPath();
        ctx.moveTo(cx + Math.sin(a) * (r - 4), cy - Math.cos(a) * (r - 4));
        ctx.lineTo(cx + Math.sin(a) * r, cy - Math.cos(a) * r); ctx.stroke();
    }
    ctx.fillStyle = isDark ? '#9CA3AF' : '#374151'; ctx.font = '7px "JetBrains Mono"'; ctx.textAlign = 'center';
    ctx.fillText('N', cx, cy - r + 8); ctx.fillText('S', cx, cy + r - 1);

    const fr = wd * Math.PI / 180, a = 15;

    ctx.strokeStyle = '#ea580c'; ctx.lineWidth = 2.5; ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(cx - Math.sin(fr) * a, cy - Math.cos(fr) * a);
    ctx.lineTo(cx + Math.sin(fr) * a, cy + Math.cos(fr) * a);
    ctx.stroke();

    const tx = cx + Math.sin(fr) * a, ty = cy + Math.cos(fr) * a;
    ctx.fillStyle = '#ea580c';
    ctx.beginPath(); ctx.arc(tx, ty, 3.5, 0, 2 * Math.PI); ctx.fill();
    ctx.fillStyle = 'rgba(255,255,255,0.2)';
    ctx.beginPath(); ctx.arc(cx, cy, 2, 0, 2 * Math.PI); ctx.fill();
}

function getSigmas(x, stab) {
    const p = PG[stab] || PG.D;
    const sy = p.ay * Math.pow(x, p.by);
    const sz = p.az * Math.pow(x, p.bz);
    return { sy, sz };
}

function gaussConc3D(x, y, z, Q, U, H, stab) {
    if (x <= 0.5) return 0;
    const { sy, sz } = getSigmas(x, stab);
    if (sy < 1e-4 || sz < 1e-4) return 0;
    const coeff = Q / (2 * Math.PI * sy * sz * Math.max(U, 0.2));
    const expY = Math.exp(-(y * y) / (2 * sy * sy));
    const expZ1 = Math.exp(-((z - H) * (z - H)) / (2 * sz * sz));
    const expZ2_correct = Math.exp(-((z + H) * (z + H)) / (2 * sz * sz));
    return coeff * expY * (expZ1 + expZ2_correct) * 1e9;
}

function heatColor(t) {
    const stops = [
        [0, 15, 23, 42],
        [0.12, 30, 41, 59],
        [0.26, 14, 116, 144],
        [0.40, 6, 182, 212],
        [0.54, 16, 185, 129],
        [0.68, 132, 204, 22],
        [0.80, 245, 158, 11],
        [0.91, 239, 68, 68],
        [1.0, 225, 29, 72],
    ];
    t = Math.max(0, Math.min(1, t));
    let i = 0;
    while (i < stops.length - 2 && t > stops[i + 1][0]) i++;
    const lo = stops[i], hi = stops[i + 1], f = (t - lo[0]) / (hi[0] - lo[0]);
    return [Math.round(lo[1] + f * (hi[1] - lo[1])),
    Math.round(lo[2] + f * (hi[2] - lo[2])),
    Math.round(lo[3] + f * (hi[3] - lo[3]))];
}

function recomputePlume() {
    const canvas = document.getElementById('plumeCanvas');
    const wrap = document.getElementById('mapWrap');
    const CW = wrap.clientWidth, CH = wrap.clientHeight;
    canvas.width = CW; canvas.height = CH;

    const Q = parseFloat(document.getElementById('Q').value);
    const H = parseFloat(document.getElementById('Heff').value);
    const stab = document.getElementById('stability').value;
    const opac = parseFloat(document.getElementById('opacity').value);
    const zReceptor = parseFloat(document.getElementById('Zslice').value);

    const maxRangeM = parseFloat((document.getElementById('maxRange') || { value: '1000' }).value) || 1000;
    const U = Math.max(W.U, 0.2);
    const WD = W.WD;

    const plumeTowardDeg = (WD + 180) % 360;
    const pRad = plumeTowardDeg * Math.PI / 180;

    const wUx = Math.sin(pRad);
    const wUy = Math.cos(pRad);
    const wPx = -wUy;
    const wPy = wUx;

    const srcPt = map.latLngToContainerPoint([SRC.lat, SRC.lon]);
    const mpp = 40075016.686 * Math.cos(SRC.lat * Math.PI / 180) / Math.pow(2, map.getZoom() + 8);

    const SCALE = 2;
    const NW = Math.ceil(CW / SCALE);
    const NH = Math.ceil(CH / SCALE);
    const grid = new Float32Array(NW * NH);
    let maxC = 0;
    let peakX = 0, peakY = 0;

    for (let py = 0; py < NH; py++) {
        for (let px = 0; px < NW; px++) {
            const dx = (px * SCALE) - srcPt.x;
            const dy = (py * SCALE) - srcPt.y;
            const dxM = dx * mpp;
            const dyM = dy * mpp;
            const xw = dxM * wUx + dyM * wUy;
            const yw = dxM * wPx + dyM * wPy;
            const c = (xw > 0 && xw <= maxRangeM) ? gaussConc3D(xw, yw, zReceptor, Q, U, H, stab) : 0;
            grid[py * NW + px] = c;
            if (c > maxC) { maxC = c; peakX = xw; peakY = yw; }
        }
    }

    plumeCache = { grid, NW, NH, SCALE, maxC, srcPt, mpp, wUx, wUy, wPx, wPy };

    const imgData = new ImageData(NW, NH);
    const logMax = Math.log10(Math.max(maxC, 1e-3));
    const logMin = logMax - 5;

    for (let i = 0; i < NW * NH; i++) {
        const c = grid[i];
        if (c < 1e-10) { imgData.data[i * 4 + 3] = 0; continue; }
        const t = Math.max(0, Math.min(1, (Math.log10(c) - logMin) / (logMax - logMin)));
        const [r, g, b] = heatColor(t);
        imgData.data[i * 4] = r;
        imgData.data[i * 4 + 1] = g;
        imgData.data[i * 4 + 2] = b;
        imgData.data[i * 4 + 3] = Math.round(t * 255 * opac);
    }

    const off = new OffscreenCanvas(NW, NH);
    off.getContext('2d').putImageData(imgData, 0, 0);
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, CW, CH);
    ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(off, 0, 0, CW, CH);

    const stP = PG[stab] || PG.D;
    const xMax = H > 0 ? Math.pow(H / (Math.sqrt(2) * stP.az), 1 / stP.bz) : 200;
    const { sy: syMax, sz: szMax } = getSigmas(Math.max(xMax, 1), stab);

    document.getElementById('sPeak').textContent = maxC.toFixed(3) + ' µg/m³';
    document.getElementById('stX').textContent = peakX > 0 ? peakX.toFixed(0) + ' m' : '—';
    document.getElementById('stY').textContent = peakY.toFixed(0) + ' m';
    document.getElementById('stZ').textContent = zReceptor + ' m';
    document.getElementById('stH').textContent = H.toFixed(1) + ' m';

    document.getElementById('axX').textContent = peakX > 0 ? peakX.toFixed(0) + ' m' : '—';
    document.getElementById('axY').textContent = peakY.toFixed(0) + ' m';
    document.getElementById('axZ').textContent = zReceptor + ' m AGL';
    document.getElementById('sSy').textContent = syMax.toFixed(1) + ' m';
    document.getElementById('sSz').textContent = szMax.toFixed(1) + ' m';
    document.getElementById('sH').textContent = H.toFixed(1) + ' m';
    document.getElementById('sDist').textContent = xMax.toFixed(0) + ' m';

    document.getElementById('scMax').textContent = maxC.toFixed(2);
    document.getElementById('scT3').textContent = (maxC * 0.75).toFixed(2);
    document.getElementById('scT2').textContent = (maxC * 0.5).toFixed(2);
    document.getElementById('scT1').textContent = (maxC * 0.25).toFixed(2);
}

map.on('mousemove', function (e) {
    const pt = map.latLngToContainerPoint(e.latlng);
    document.getElementById('coordTxt').textContent =
        e.latlng.lat.toFixed(5) + '°N  ' + e.latlng.lng.toFixed(5) + '°E';

    if (!plumeCache) return;
    const d = plumeCache;
    const px = Math.floor(pt.x / d.SCALE);
    const py = Math.floor(pt.y / d.SCALE);
    if (px >= 0 && px < d.NW && py >= 0 && py < d.NH) {
        const c = d.grid[py * d.NW + px];
        const dx = pt.x - d.srcPt.x, dy = pt.y - d.srcPt.y;
        const dxM = dx * d.mpp, dyM = dy * d.mpp;
        const xw = dxM * d.wUx + dyM * d.wUy;
        const yw = dxM * d.wPx + dyM * d.wPy;
        const zw = parseFloat(document.getElementById('Zslice').value);
        const tip = document.getElementById('tooltip');
        tip.style.display = 'block';
        tip.style.left = (pt.x + 16) + 'px';
        tip.style.top = (pt.y - 12) + 'px';
        tip.innerHTML = 'x=<em>' + xw.toFixed(0) + 'm</em> y=<em>' + yw.toFixed(0) +
            'm</em> z=<em>' + zw + 'm</em><br>Density=<em>' + c.toFixed(4) + ' µg/m³</em>';
    }
});
map.on('mouseout', () => { document.getElementById('tooltip').style.display = 'none'; });
map.on('moveend zoomend', recomputePlume);

function bindSlider(id, lid, fmt) {
    const el = document.getElementById(id);
    const ll = document.getElementById(lid);
    el.addEventListener('input', () => { ll.textContent = fmt ? fmt(+el.value) : el.value; });
    el.addEventListener('change', recomputePlume);
}
bindSlider('Q', 'lQ', v => v.toFixed(5));
bindSlider('maxRange', 'lRange');
bindSlider('Heff', 'lH');
bindSlider('Zslice', 'lZslice');
bindSlider('opacity', 'lopac');

const stabNames = { A: 'Unstable A', B: 'Unstable B', C: 'Unstable C', D: 'Near Neutral', E: 'Stable E', F: 'Stable F' };
document.getElementById('stability').addEventListener('change', function () {
    document.getElementById('lsc').textContent = stabNames[this.value] || this.value;
    recomputePlume();
});

document.getElementById('inAlt').addEventListener('input', function () {
    document.getElementById('Heff').value = this.value;
    document.getElementById('lH').textContent = this.value;
});

let currentTheme = 'dark'; // Starting theme

function toggleTheme() {
    currentTheme = currentTheme === 'light' ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', currentTheme);

    const btn = document.getElementById('btnTheme');
    if (currentTheme === 'dark') {
        btn.innerHTML = '☀️ Light Mode';
        btn.style.borderColor = 'var(--border)';
        btn.style.color = 'var(--text-heading)';
    } else {
        btn.innerHTML = '🌙 Dark Mode';
        btn.style.borderColor = 'var(--accent)';
        btn.style.color = 'var(--accent)';
    }
    localStorage.setItem('theme', currentTheme);
    if (typeof W !== 'undefined' && W.WD !== undefined) drawCompass(W.WD);
}
window.toggleTheme = toggleTheme;

function initTheme() {
    const savedTheme = localStorage.getItem('theme') || 'dark';
    if (savedTheme === 'light') {
        currentTheme = 'dark'; // Trick it so the toggle triggers correctly
        toggleTheme();
    }
}

window.addEventListener('load', () => {
    initTheme();
    updateDroneHUD();
    drawCompass(315);
    fetchLiveWeather();

    setInterval(() => {
        document.getElementById('hudTime').textContent =
            'System Sync · ' + new Date().toLocaleTimeString('en-BD', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    }, 1000);

    setInterval(() => {
        if (windMode === 'live') fetchLiveWeather();
    }, 5 * 60 * 1000);

    setTimeout(() => { recomputePlume(); }, 1000);
});