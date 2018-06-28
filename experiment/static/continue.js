function Continue(event){
	if(event.keyCode == 32){
		nextPage();
	}
}
document.addEventListener("keydown", Continue);