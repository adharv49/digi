document.addEventListener('DOMContentLoaded', function () {
    var accordions = document.querySelectorAll('.accordion');
    accordions.forEach(function (accordion) {
        var header = accordion.querySelector('.accordion-header');
        header.addEventListener('click', function () {
            accordion.classList.toggle('open');
        });
    });

    document.getElementById('src_image').addEventListener('change', function (event) {
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function (e) {
                document.getElementById('src_preview').src = e.target.result;
            };
            reader.readAsDataURL(file);
        }
    });

    document.getElementById('ref_image').addEventListener('change', function (event) {
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function (e) {
                document.getElementById('ref_preview').src = e.target.result;
            };
            reader.readAsDataURL(file);
        }
    });
});

