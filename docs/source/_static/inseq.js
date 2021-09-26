function addIcon() {
    const inseqLogo = "_static/inseq_logo.png";
    const image = document.createElement("img");
    image.setAttribute("src", inseqLogo);

    const div = document.createElement("div");
    div.appendChild(image);
    div.style.textAlign = 'center';
    div.style.paddingTop = '30px';

    const scrollDiv = document.querySelector(".wy-side-scroll");
    scrollDiv.prepend(div);
}

function addCustomFooter() {
    const customFooter = document.createElement("div");
    customFooter.classList.add("footer");

    const social = document.createElement("div");
    social.classList.add("footer__Social");

    const imageDetails = [
        { link: "https://gsarti.com", imageLink: "https://huggingface.co/landing/assets/transformers-docs/website.svg" },
        { link: "https://twitter.com/gsarti_", imageLink: "https://huggingface.co/landing/assets/transformers-docs/twitter.svg" },
        { link: "https://github.com/gsarti", imageLink: "https://huggingface.co/landing/assets/transformers-docs/github.svg" },
        { link: "https://www.linkedin.com/gabrielesarti/", imageLink: "https://huggingface.co/landing/assets/transformers-docs/linkedin.svg" }
    ];

    imageDetails.forEach(imageLinks => {
        const link = document.createElement("a");
        const image = document.createElement("img");
        image.src = imageLinks.imageLink;
        link.href = imageLinks.link;
        image.style.width = "30px";
        image.classList.add("footer__CustomImage");
        link.appendChild(image);
        social.appendChild(link);
    });

    customFooter.appendChild(social);
    document.querySelector("footer").appendChild(customFooter);
}


function onLoad() {
    addIcon();
    addCustomFooter();
}

window.addEventListener("load", onLoad);
