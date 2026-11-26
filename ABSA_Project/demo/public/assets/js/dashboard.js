// Biểu đồ dashboard BHI. Dữ liệu do PHP nhúng dưới dạng JSON.
(function () {
    'use strict';

    var dataElement = document.getElementById('bhi-chart-data');
    if (!dataElement || typeof window.Chart === 'undefined') {
        return;
    }

    var data;
    try {
        data = JSON.parse(dataElement.textContent);
    } catch (error) {
        return;
    }

    var numberFormat = new Intl.NumberFormat('vi-VN', { maximumFractionDigits: 1 });
    var Chart = window.Chart;

    Chart.defaults.font.family = '"Segoe UI", system-ui, -apple-system, Arial, sans-serif';
    Chart.defaults.color = '#4a5c57';

    function scoreColor(score) {
        if (score >= 70) { return '#0f766e'; }
        if (score >= 50) { return '#d97706'; }
        return '#dc2626';
    }

    // ------------------------------------------------------------------
    // Điểm theo khía cạnh
    // ------------------------------------------------------------------
    var aspectCanvas = document.getElementById('aspectChart');
    if (aspectCanvas && data.aspects && data.aspects.labels.length > 0) {
        new Chart(aspectCanvas, {
            type: 'bar',
            data: {
                labels: data.aspects.labels,
                datasets: [{
                    label: 'Điểm',
                    data: data.aspects.scores,
                    backgroundColor: data.aspects.scores.map(scoreColor),
                    borderRadius: 6,
                    maxBarThickness: 48
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        ticks: { callback: function (value) { return value + ' điểm'; } },
                        grid: { color: 'rgba(16, 35, 31, .07)' }
                    },
                    x: { grid: { display: false } }
                },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: function (context) {
                                return ' ' + numberFormat.format(context.parsed.y) + '/100';
                            }
                        }
                    }
                }
            }
        });
    }

    // ------------------------------------------------------------------
    // Phân bố cảm xúc
    // ------------------------------------------------------------------
    var sentimentCanvas = document.getElementById('sentimentChart');
    if (sentimentCanvas && data.sentiment) {
        var sentimentValues = data.sentiment.values.map(Number);
        var sentimentTotal = sentimentValues.reduce(function (sum, value) { return sum + value; }, 0);

        if (sentimentTotal > 0) {
            new Chart(sentimentCanvas, {
                type: 'doughnut',
                data: {
                    labels: data.sentiment.labels,
                    datasets: [{
                        data: sentimentValues,
                        backgroundColor: data.sentiment.colors,
                        borderColor: '#ffffff',
                        borderWidth: 3,
                        hoverOffset: 4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    cutout: '62%',
                    plugins: {
                        legend: { position: 'bottom', labels: { usePointStyle: true, boxWidth: 9 } },
                        tooltip: {
                            callbacks: {
                                label: function (context) {
                                    var value = Number(context.parsed) || 0;
                                    var percent = sentimentTotal > 0 ? (value / sentimentTotal * 100) : 0;
                                    return ' ' + context.label + ': ' + value + ' (' + numberFormat.format(percent) + '%)';
                                }
                            }
                        }
                    }
                }
            });
        }
    }

    // ------------------------------------------------------------------
    // Xu hướng BHI theo tháng
    // ------------------------------------------------------------------
    var trendCanvas = document.getElementById('trendChart');
    if (trendCanvas && data.trend && data.trend.labels.length > 0) {
        new Chart(trendCanvas, {
            type: 'line',
            data: {
                labels: data.trend.labels,
                datasets: [{
                    label: 'BHI',
                    data: data.trend.values,
                    borderColor: '#0f766e',
                    backgroundColor: 'rgba(15, 118, 110, .12)',
                    pointBackgroundColor: '#0f766e',
                    pointRadius: 4,
                    pointHoverRadius: 6,
                    fill: true,
                    tension: 0.3,
                    spanGaps: false
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        ticks: { callback: function (value) { return value + ' điểm'; } },
                        grid: { color: 'rgba(16, 35, 31, .07)' }
                    },
                    x: { grid: { display: false } }
                },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: function (context) {
                                return ' BHI: ' + numberFormat.format(context.parsed.y) + '/100';
                            }
                        }
                    }
                }
            }
        });
    }
})();