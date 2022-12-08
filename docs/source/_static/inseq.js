function addIcon() {
    const inseqLogo = "/_static/inseq_logo.png";
    const link = document.createElement("a");
    const indexpath = "/index.html";
    link.setAttribute("href", indexpath);
    const image = document.createElement("img");
    image.setAttribute("src", inseqLogo);
    image.setAttribute("alt", "Inseq logo");
    image.style.width = "80%"
    image.style.paddingRight = "10%";
    link.appendChild(image);
    const desc = document.createElement("p");
    desc.innerHTML = "Interpretability for sequence-to-sequence models 🔍";
    desc.style.color = "black";
    desc.style.paddingLeft = "20px";
    desc.style.paddingTop = "10px";
    desc.style.width = "80%"

    const div = document.createElement("div");
    div.appendChild(link);
    div.appendChild(desc);
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
        { link: "https://interpretingdl.github.io", imageLink: "https://huggingface.co/landing/assets/transformers-docs/website.svg" },
        //{ link: "https://twitter.com/inseq", imageLink: "https://huggingface.co/landing/assets/transformers-docs/twitter.svg" },
        { link: "https://github.com/inseq-team", imageLink: "https://huggingface.co/landing/assets/transformers-docs/github.svg" },
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

function resizeHtmlExamples() {
    var examples = document.getElementsByClassName("html-example");
    for (const ex of examples) {
        const iframe = ex.firstElementChild;
        const zoom = iframe.getAttribute("scale")
        const origHeight = iframe.contentWindow.document.body.scrollHeight
        const origWidth = iframe.contentWindow.document.body.scrollWidth
        ex.style.height = ((origHeight * zoom) + 50) + "px";
        const frameHeight = origHeight / zoom
        const frameWidth = origWidth / zoom
        // add extra 50 pixels - in reality need just a bit more
        iframe.style.height = frameHeight + "px"
        // set the width of the iframe as the width of the iframe content
        iframe.style.width = frameWidth + 'px';
        iframe.style.zoom = zoom;
        iframe.style.MozTransform = `scale(${zoom})`;
        iframe.style.WebkitTransform = `scale(${zoom})`;
        iframe.style.transform = `scale(${zoom})`;
        iframe.style.MozTransformOrigin = "0 0";
        iframe.style.WebkitTransformOrigin = "0 0";
        iframe.style.transformOrigin = "0 0";
    }
}

function onLoad() {
    addIcon();
    addCustomFooter();
    resizeHtmlExamples();
}

window.addEventListener("load", onLoad);
window.onresize = function() {
    window.location.href = window.location.href;
}
