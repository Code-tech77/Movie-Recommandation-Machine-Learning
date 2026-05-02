function filterMovies() {
    let input = document.getElementById("searchBox").value.toLowerCase();
    let select = document.getElementById("movieDropdown");
    let options = select.options;

    for (let i = 0; i < options.length; i++) {
        let txt = options[i].text.toLowerCase();
        options[i].style.display = txt.includes(input) ? "" : "none";
    }
}