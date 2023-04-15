function secondLevel(beginValue)
{
	if (beginValue == "null")
	{
		document.getElementById("suggestResult").style.display = "none";
		document.getElementById("predictResult").style.display = "none";
		document.getElementById("compareResult").style.display = "none";

	}
	if (beginValue == "suggest")
	{
		document.getElementById("suggestResult").style.display = "inline-block";
		document.getElementById("predictResult").style.display = "none";
		document.getElementById("compareResult").style.display = "none";
	}
	if (beginValue == "predict")
	{
		document.getElementById("suggestResult").style.display = "none";
		document.getElementById("predictResult").style.display = "inline-block";
		document.getElementById("compareResult").style.display = "none";
	}
	if (beginValue == "compare")
	{
		document.getElementById("suggestResult").style.display = "none";
		document.getElementById("predictResult").style.display = "none";
		document.getElementById("compareResult").style.display = "inline-block";
	}
}