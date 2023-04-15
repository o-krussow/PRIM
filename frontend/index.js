function secondLevel(beginValue)
{
	if (beginValue == "suggest")
	{
		document.getElementById("suggestResult").style.display = "inline-block";
		document.getElementById("predictResult").style.display = "none";
		document.getElementById("compareResult").style.display = "none";
		document.getElementById("sentence").innerHTML = "suggest";
		showData('suggestSubmit');
	}
	if (beginValue == "predict")
	{
		document.getElementById("suggestResult").style.display = "none";
		document.getElementById("predictResult").style.display = "inline-block";
		document.getElementById("compareResult").style.display = "none";
		document.getElementById("sentence").innerHTML = "predict";
		document.getElementById("suggestData").style.display = "none";
		document.getElementById("predictData").style.display = "none";
		document.getElementById("compareData").style.display = "none";
	}
	if (beginValue == "compare")
	{
		document.getElementById("suggestResult").style.display = "none";
		document.getElementById("predictResult").style.display = "none";
		document.getElementById("compareResult").style.display = "inline-block";
		document.getElementById("sentence").innerHTML = "compare";
		document.getElementById("suggestData").style.display = "none";
		document.getElementById("predictData").style.display = "none";
		document.getElementById("compareData").style.display = "none";
	}
}

function showData(whichData)
{
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
		let stock = document.getElementById("stockInput").value;
		let time = document.getElementById("timeInput").value;
		if (stock.length < 1 || time.length < 1)
		{
			alert("Field(s) are blank");
			return;
		}
		document.getElementById("suggestData").style.display = "none";
		document.getElementById("predictData").style.display = "block";
		document.getElementById("compareData").style.display = "none";
	}
	if (whichData == 'compareSubmit')
	{
		/* INSERT CODE TO GENERATE STOCK DATA FOR COMPARE HERE */
		let stockOne = document.getElementById("stockOne").value;
		let stockTwo = document.getElementById("stockTwo").value;
		if (stockOne.length < 1 || stockTwo.length < 1)
		{
			alert("Field(s) are blank");
			return;
		}
		document.getElementById("suggestData").style.display = "none";
		document.getElementById("predictData").style.display = "none";
		document.getElementById("compareData").style.display = "block";
	}
}

function active(tab)
{
	if (tab == 1)
	{
		document.getElementById("one").classList.add("active");
		document.getElementById("two").classList.remove("active");
		document.getElementById("optionOne").style.display = "inline-block";

	}
	if (tab == 2)
	{
		document.getElementById("one").classList.remove("active");
		document.getElementById("two").classList.add("active");
		document.getElementById("optionOne").style.display = "none";
	}
}