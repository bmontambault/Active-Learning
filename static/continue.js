function Continue(event){
	if(event.keyCode == 32){
		document.getElementById("form").submit();
	}
}
document.addEventListener("keydown", Continue);