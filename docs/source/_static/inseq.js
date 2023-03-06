var curr_width = $(window).width();

function resizeHtmlExamples() {
    var examples = document.getElementsByClassName("html-example");
    for (const ex of examples) {
        const iframe = ex.firstElementChild;
        const zoom = iframe.getAttribute("scale")
        ex.style.height = ((iframe.contentWindow.document.body.scrollHeight - 50) * zoom) + "px";
        iframe.style.height = (iframe.contentWindow.document.body.scrollHeight - 50) + "px"
        iframe.style.width = "133%";
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
    resizeHtmlExamples();
}

window.addEventListener("load", onLoad);
window.onresize = function() {
    var wwidth = $(window).width();
    if( curr_width !== wwidth ){
        window.location.reload();
        curr_width = wwidth;
    }
}
