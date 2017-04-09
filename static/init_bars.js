function init_bars(selected){
	var allBars = document.getElementsByClassName("bar")
	for(var i=0; i<allBars.length; i++){
		var ctx=allBars[i].getContext("2d")
		ctx.fillStyle = "grey"
		ctx.fillRect(0, 0, allBars[i].width, allBars[i].height)
	}
	var selectedBars = document.getElementsByClassName("selected")
	for(var i=0; i<selectedBars.length; i++){
		var ctx=selectedBars[i].getContext("2d")
		ctx.fillStyle = "black"
		ctx.clearRect(0, 0, selectedBars[i].width, selectedBars[i].height);
		ctx.fillRect(0, selectedBars[i].height-selected[i], selectedBars[i].width, selectedBars[i].height)
	}
}