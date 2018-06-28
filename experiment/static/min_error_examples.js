var example1Bars = document.getElementsByClassName("example1")
for(var i=0; i<example1Bars.length; i++){
	var ctx = example1Bars[i].getContext("2d")
	ctx.fillStyle = "black"
	ctx.clearRect(0, 0, example1Bars[i].width, example1Bars[i].height);
	ctx.fillRect(0, example1Bars[i].height-100, example1Bars[i].width, example1Bars[i].height)
	ctx.fillStyle = "red"
	ctx.fillRect(0, example1Bars[i].height-50, example1Bars[i].width, example1Bars[i].height)
}

var example2Bars = document.getElementsByClassName("example2")
for(var i=0; i<example2Bars.length; i++){
	var ctx = example2Bars[i].getContext("2d")
	ctx.fillStyle = "red"
	ctx.clearRect(0, 0, example2Bars[i].width, example2Bars[i].height);
	ctx.fillRect(0, example2Bars[i].height-200, example2Bars[i].width, example2Bars[i].height)
	ctx.fillStyle = "black"
	ctx.fillRect(0, example2Bars[i].height-120, example2Bars[i].width, example2Bars[i].height)
}