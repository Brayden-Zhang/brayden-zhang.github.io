$(document).ready(function() {

    var options = {
      strings: ['Programmer', 'Builder', 'Engineering Science Student @ UToronto'],
      typeSpeed: 60,
      loop:true
    };

    var typed = new Typed('.element', options);

    $(window).scroll(function() {
        if ($(this).scrollTop() > 100) {
            $('.header-area').addClass('sticky');
        } else {
            $('.header-area').removeClass('sticky');
        }
    });

    ScrollReveal({
        distance: '100px',
        duration:2000,
        delay:200
    });

    ScrollReveal().reveal('.header a, .banner-content, .about-content',  {origin: 'left'});
    ScrollReveal().reveal('.header ul, .banner-img, .about-skills', {origin: 'right'});
    ScrollReveal().reveal('.section-title', {origin: 'top'});
    ScrollReveal().reveal('.projects, .contact, .footer-area', {origin: 'bottom'});

    let menu_icon = document.querySelector('.menu-icon i')
    let menu_list = document.querySelector('.header ul')

    menu_icon.onClick = () => {
        menu_icon.classList.toggle('bx-x');
        menu_list.classList.toggle('show');
    }
});