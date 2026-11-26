// Hành vi nhỏ dùng chung cho giao diện.
(function () {
    'use strict';

    // Xác nhận trước khi xóa.
    document.querySelectorAll('[data-confirm]').forEach(function (form) {
        form.addEventListener('submit', function (event) {
            var message = form.getAttribute('data-confirm') || 'Bạn có chắc chắn?';
            if (!window.confirm(message)) {
                event.preventDefault();
            }
        });
    });

    // Đếm ký tự cho textarea.
    document.querySelectorAll('[data-counter]').forEach(function (input) {
        var target = document.querySelector(input.getAttribute('data-counter'));
        if (!target) return;

        var max = input.getAttribute('maxlength') || '';
        var update = function () {
            target.textContent = input.value.length + (max ? '/' + max : '') + ' ký tự';
        };
        input.addEventListener('input', update);
        update();
    });
})();