/* AeroVision - Chart.js Utility Functions */

function createBarChart(canvasId, labels, datasets, title, options = {}) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    return new Chart(ctx, {
        type: 'bar',
        data: { labels: labels, datasets: datasets },
        options: {
            responsive: true,
            plugins: {
                title: { display: !!title, text: title }
            },
            ...options
        }
    });
}

function createLineChart(canvasId, labels, data, title, options = {}) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Average AQI',
                data: data,
                borderColor: '#0d6efd',
                backgroundColor: 'rgba(13,110,253,0.1)',
                tension: 0.3,
                fill: true
            }]
        },
        options: {
            responsive: true,
            plugins: {
                title: { display: !!title, text: title }
            },
            ...options
        }
    });
}

function getAqiColor(aqi) {
    if (aqi <= 50) return '#00b050';
    if (aqi <= 100) return '#92d050';
    if (aqi <= 200) return '#ffff00';
    if (aqi <= 300) return '#ff9900';
    if (aqi <= 400) return '#ff0000';
    return '#990000';
}
