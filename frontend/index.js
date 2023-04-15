function secondLevel(beginValue)
{
	if (beginValue == "null")
	{
		document.getElementById("suggestResult").style.display = "none";
		document.getElementById("predictResult").style.display = "none";
		document.getElementById("compareResult").style.display = "none";
		showData('nullSubmit');

	}
	if (beginValue == "suggest")
	{
		document.getElementById("suggestResult").style.display = "inline-block";
		document.getElementById("predictResult").style.display = "none";
		document.getElementById("compareResult").style.display = "none";
		showData('suggestSubmit');
	}
	if (beginValue == "predict")
	{
		document.getElementById("suggestResult").style.display = "none";
		document.getElementById("predictResult").style.display = "inline-block";
		document.getElementById("compareResult").style.display = "none";
		document.getElementById("suggestData").style.display = "none";
		document.getElementById("predictData").style.display = "none";
		document.getElementById("compareData").style.display = "none";
	}
	if (beginValue == "compare")
	{
		document.getElementById("suggestResult").style.display = "none";
		document.getElementById("predictResult").style.display = "none";
		document.getElementById("compareResult").style.display = "inline-block";
		document.getElementById("suggestData").style.display = "none";
		document.getElementById("predictData").style.display = "none";
		document.getElementById("compareData").style.display = "none";
	}
}

function showData(whichData)
{
	if (whichData == 'nullSubmit')
	{
		document.getElementById("suggestData").style.display = "none";
		document.getElementById("predictData").style.display = "none";
		document.getElementById("predictData").style.display = "none";
	}
	if (whichData == 'suggestSubmit')
	{
		/* INSERT CODE TO GENERATE STOCK DATA FOR SUGGEST HERE */
		document.getElementById("suggestData").style.display = "block";
		document.getElementById("predictData").style.display = "none";
		document.getElementById("compareData").style.display = "none";
	}
	if (whichData == 'predictSubmit')
	{
		/* INSERT CODE TO GENERATE STOCK DATA FOR PREDICT HERE */
		document.getElementById("suggestData").style.display = "none";
		document.getElementById("predictData").style.display = "block";
		document.getElementById("compareData").style.display = "none";
	}
	if (whichData == 'compareSubmit')
	{
		/* INSERT CODE TO GENERATE STOCK DATA FOR COMPARE HERE */
		document.getElementById("suggestData").style.display = "none";
		document.getElementById("predictData").style.display = "none";
		document.getElementById("compareData").style.display = "block";
	}
}