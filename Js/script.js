// ===== SCROLL LOGIC FOR MORPHING NAVBAR & SCROLLSPY =====
window.addEventListener('scroll', () => {

    // 1. Morphing Navbar Logic
    const navbar = document.querySelector('.navbar');
    if (window.scrollY > 80) {
        navbar.classList.add('scrolled');
        document.body.classList.add('nav-is-scrolled');
    } else {
        navbar.classList.remove('scrolled');
        document.body.classList.remove('nav-is-scrolled');
    }

    // 2. Active Link Scrollspy Logic
    const sections = document.querySelectorAll('#sensors, #trends, #maps');
    let current = '#sensors'; // Default to Sensors

    sections.forEach(section => {
        const rect = section.getBoundingClientRect();
        // If the section hits the top 250px of the viewport, count it as active
        if (rect.top <= 250) {
            current = '#' + section.id;
        }
    });

    // Loop through nav links and apply the active class dynamically
    document.querySelectorAll('.nav-link[href^="#"]').forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('href') === current) {
            link.classList.add('active');
        }
    });
});

// ===== CONFIGURATION =====
const GM_TUBE_API = 'https://script.google.com/macros/s/AKfycbzCBRfLEIenGJcOjMiCfhb3aIdF_kop9LOhxdMUeVzn__Sl10te8O-uQcWIacMrw4WpWw/exec';
const ENV_SENSOR_API = 'https://script.google.com/macros/s/AKfycbyp0PaEELfInYzXKDQxFC5oCGqDaUvf9LptF224IkEq3BBQpCXFol5jRGmjsADJvY50/exec';
const UPDATE_INTERVAL = 5000;

const THRESHOLDS = {
    temperature: { min: 18, max: 30 },
    humidity: { min: 30, max: 70 },
    dose: { safe: 0.30, elevated: 1.0, high: 10.0 }
};

let doseChart, envChart;
let envData = null;
let radData = null;

// ===== MAP TAB SWITCHING =====
function switchMapTab(tabName) {
    document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));
    document.querySelectorAll('.tab-button').forEach(btn => btn.classList.remove('active'));
    document.getElementById('tab-' + tabName).classList.add('active');
    event.target.classList.add('active'); // Needs to be triggered via inline or passed event
}

// Ensure the switchMapTab function is exposed to the global window object
// so the inline onclick="" handlers in HTML can trigger it.
window.switchMapTab = switchMapTab; 

// ===== DATA FETCHING =====
async function fetchEnvData() {
    try {
        const response = await fetch(ENV_SENSOR_API);
        const data = await response.json();
        if (data.error) throw new Error(data.error);
        return data;
    } catch (error) {
        console.error('Error fetching environmental data:', error);
        showError('ATMOSPHERIC NODE UNREACHABLE');
        return null;
    }
}

async function fetchRadData() {
    try {
        const response = await fetch(GM_TUBE_API);
        const data = await response.json();
        if (data.error) throw new Error(data.error);
        return data;
    } catch (error) {
        console.error('Error fetching radiation data:', error);
        showError('GEIGER NODE UNREACHABLE');
        return null;
    }
}

function showError(message) {
    const errorEl = document.getElementById('errorMessage');
    errorEl.textContent = '[FAULT] ' + message;
    errorEl.style.display = 'block';
    setTimeout(() => errorEl.style.display = 'none', 10000);
}

// ===== UPDATE DASHBOARD =====
function updateDashboard() {
    if (!envData && !radData) return;

    document.getElementById('loading').style.display = 'none';
    document.getElementById('dashboard').style.display = 'block';

    if (envData && envData.latest) {
        const env = envData.latest;
        document.getElementById('temp').textContent = env.temperature + '°C';
        updateStatus('temp-status', env.temperature, THRESHOLDS.temperature.min, THRESHOLDS.temperature.max);
        document.getElementById('humidity').textContent = env.humidity + '%';
        updateStatus('humidity-status', env.humidity, THRESHOLDS.humidity.min, THRESHOLDS.humidity.max);
        document.getElementById('h2-raw').textContent = env.h2_raw;
        document.getElementById('h2-voltage').textContent = env.h2_voltage + 'V';
        document.getElementById('h2-do').textContent = env.h2_do;

        const h2Status = document.getElementById('h2-status');
        if (env.h2_do == 1) {
            h2Status.textContent = 'SYS. NOMINAL';
            h2Status.className = 'status-badge safe';
        } else {
            h2Status.textContent = 'CONTAMINATION DETECTED';
            h2Status.className = 'status-badge danger';
            showAlert('HYDROGEN GAS CONCENTRATION EXCEEDS TOLERANCE');
        }

        document.getElementById('envUpdate').textContent = new Date(env.timestamp).toLocaleTimeString();
        updateEnvChart(envData.history);
    }

    if (radData && radData.latest) {
        const rad = radData.latest;
        document.getElementById('dose').textContent = rad.dose.toFixed(3) + ' µSv/h';
        document.getElementById('cpm').textContent = rad.cpm;
        document.getElementById('cps').textContent = rad.cps;

        document.getElementById('rad-text').textContent = rad.status.replace(/[\u{1F300}-\u{1F9FF}]/gu, '').trim();

        const radStatus = document.getElementById('rad-status');
        if (rad.dose < THRESHOLDS.dose.safe) {
            radStatus.textContent = 'SYS. NOMINAL';
            radStatus.className = 'status-badge safe';
        } else if (rad.dose < THRESHOLDS.dose.elevated) {
            radStatus.textContent = 'ELEVATED LEVEL';
            radStatus.className = 'status-badge warning';
        } else if (rad.dose < THRESHOLDS.dose.high) {
            radStatus.textContent = 'WARNING';
            radStatus.className = 'status-badge warning';
            showAlert('RADIATION LEVELS EXCEED NOMINAL PARAMETERS');
        } else {
            radStatus.textContent = 'CRITICAL ALERT';
            radStatus.className = 'status-badge danger';
            showAlert('CRITICAL RADIATION EXPOSURE DETECTED');
        }

        document.getElementById('radUpdate').textContent = new Date(rad.timestamp).toLocaleTimeString();
        updateDoseChart(radData.history);
    }
}

function updateStatus(elementId, value, min, max) {
    const element = document.getElementById(elementId);
    if (value < min || value > max) {
        element.textContent = 'DEVIATION DETECTED';
        element.className = 'status-badge warning';
    } else {
        element.textContent = 'SYS. NOMINAL';
        element.className = 'status-badge safe';
    }
}

function showAlert(message) {
    const banner = document.getElementById('alertBanner');
    banner.textContent = '[CRITICAL ALERT] ' + message;
    banner.style.display = 'block';
    setTimeout(() => banner.style.display = 'none', 10000);
}

// ===== CHARTS =====
function updateDoseChart(history) {
    if (!history || history.length === 0) return;
    const timestamps = history.map(row => new Date(row.timestamp).toLocaleTimeString());
    const doses = history.map(row => row.dose);

    const isDark = currentTheme === 'dark';
    const textColor = isDark ? '#E5E7EB' : '#0F172A';
    const gridColor = isDark ? 'rgba(243,244,246,0.1)' : 'rgba(15,23,42,0.1)';

    if (doseChart) doseChart.destroy();
    const doseCtx = document.getElementById('doseChart').getContext('2d');
    doseChart = new Chart(doseCtx, {
        type: 'line',
        data: {
            labels: timestamps,
            datasets: [{
                label: 'Dose Rate (µSv/h)',
                data: doses,
                borderColor: '#DC2626',
                backgroundColor: 'rgba(220, 38, 38, 0.1)',
                borderWidth: 2,
                tension: 0.2,
                fill: true,
                pointRadius: 1,
                pointHoverRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        color: textColor,
                        font: { family: "'Geist Mono', monospace", size: 11 },
                        usePointStyle: true,
                        padding: 20
                    }
                }
            },
            scales: {
                x: { ticks: { color: textColor, font: { family: "'Geist Mono', monospace", size: 10 } }, grid: { color: gridColor } },
                y: { ticks: { color: textColor, font: { family: "'Geist Mono', monospace", size: 10 } }, grid: { color: gridColor } }
            }
        }
    });
}

function updateEnvChart(history) {
    if (!history || history.length === 0) return;
    const timestamps = history.map(row => new Date(row.timestamp).toLocaleTimeString());
    const temps = history.map(row => row.temperature);
    const humidities = history.map(row => row.humidity);

    const isDark = currentTheme === 'dark';
    const textColor = isDark ? '#E5E7EB' : '#0F172A';
    const gridColor = isDark ? 'rgba(243,244,246,0.1)' : 'rgba(15,23,42,0.1)';

    if (envChart) envChart.destroy();
    const envCtx = document.getElementById('envChart').getContext('2d');
    envChart = new Chart(envCtx, {
        type: 'line',
        data: {
            labels: timestamps,
            datasets: [
                {
                    label: 'Temp (°C)',
                    data: temps,
                    borderColor: '#2563EB',
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    tension: 0.2,
                    yAxisID: 'y',
                    pointRadius: 1,
                    pointHoverRadius: 4
                },
                {
                    label: 'Humidity (%)',
                    data: humidities,
                    borderColor: '#059669',
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    tension: 0.2,
                    yAxisID: 'y1',
                    pointRadius: 1,
                    pointHoverRadius: 4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        color: textColor,
                        font: { family: "'Geist Mono', monospace", size: 11 },
                        usePointStyle: true,
                        padding: 20
                    }
                }
            },
            scales: {
                x: { ticks: { color: textColor, font: { family: "'Geist Mono', monospace", size: 10 } }, grid: { color: gridColor } },
                y: {
                    type: 'linear',
                    position: 'left',
                    ticks: { color: textColor, font: { family: "'Geist Mono', monospace", size: 10 } },
                    grid: { color: gridColor }
                },
                y1: {
                    type: 'linear',
                    position: 'right',
                    ticks: { color: textColor, font: { family: "'Geist Mono', monospace", size: 10 } },
                    grid: { display: false }
                }
            }
        }
    });
}

async function fetchAllData() {
    const [env, rad] = await Promise.all([fetchEnvData(), fetchRadData()]);
    if (env) envData = env;
    if (rad) radData = rad;
    updateDashboard();
}

async function init() {
    initTheme();
    await fetchAllData();
}

// ===== NAVIGATION =====
function toggleMenu() {
    const navMenu = document.querySelector('.nav-menu');
    const navToggle = document.querySelector('.nav-toggle');
    navMenu.classList.toggle('active');
    navToggle.classList.toggle('active');
}
window.toggleMenu = toggleMenu;

// ===== THEME TOGGLE =====
let currentTheme = localStorage.getItem('theme') || 'dark';

function toggleTheme() {
    currentTheme = currentTheme === 'light' ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', currentTheme);

    const themeText = document.getElementById('themeText');
    themeText.textContent = currentTheme === 'dark' ? 'MODE: LIGHT' : 'MODE: DARK';

    localStorage.setItem('theme', currentTheme);

    if (doseChart || envChart) {
        updateChartsForTheme();
    }
}
window.toggleTheme = toggleTheme;

function initTheme() {
    document.documentElement.setAttribute('data-theme', currentTheme);
    const themeText = document.getElementById('themeText');
    if(themeText) {
        themeText.textContent = currentTheme === 'dark' ? 'MODE: LIGHT' : 'MODE: DARK';
    }
}

function updateChartsForTheme() {
    const isDark = currentTheme === 'dark';
    const textColor = isDark ? '#E5E7EB' : '#0F172A';
    const gridColor = isDark ? 'rgba(243,244,246,0.1)' : 'rgba(15,23,42,0.1)';

    if (doseChart) {
        doseChart.options.plugins.legend.labels.color = textColor;
        doseChart.options.scales.x.ticks.color = textColor;
        doseChart.options.scales.x.grid.color = gridColor;
        doseChart.options.scales.y.ticks.color = textColor;
        doseChart.options.scales.y.grid.color = gridColor;
        doseChart.update();
    }

    if (envChart) {
        envChart.options.plugins.legend.labels.color = textColor;
        envChart.options.scales.x.ticks.color = textColor;
        envChart.options.scales.x.grid.color = gridColor;
        envChart.options.scales.y.ticks.color = textColor;
        envChart.options.scales.y.grid.color = gridColor;
        envChart.options.scales.y1.ticks.color = textColor;
        envChart.update();
    }
}

document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        if (this.getAttribute('href') !== '#') {
            e.preventDefault();
            const targetId = this.getAttribute('href');
            const targetSection = document.querySelector(targetId);
            if (targetSection) {
                targetSection.scrollIntoView({ behavior: 'smooth' });
                if (window.innerWidth <= 900) {
                    toggleMenu(); // Close mobile menu on click
                }
            }
        }
    });
});

setInterval(fetchAllData, UPDATE_INTERVAL);
init();