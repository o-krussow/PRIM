function secondLevel(beginValue)
{
	if (beginValue == "suggest")
	{
		document.getElementById("suggestResult").style.display = "inline-block";
		document.getElementById("predictResult").style.display = "none";
		document.getElementById("compareResult").style.display = "none";
		document.getElementById("formTrain").style.display = "none";
		document.getElementById("sentence").innerHTML = "suggest";
		showData('suggestSubmit');
	}
	if (beginValue == "predict")
	{
		document.getElementById("suggestResult").style.display = "none";
		document.getElementById("predictResult").style.display = "inline-block";
		document.getElementById("compareResult").style.display = "none";
		document.getElementById("formTrain").style.display = "none";
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
		document.getElementById("formTrain").style.display = "block";
		document.getElementById("sentence").innerHTML = "train";
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
		let stock = document.getElementById("stockInput").value;
		let time = document.getElementById("timeInput").value;
		/* TAKE VALUES INPUTTED AND MAKE THEM SUITABLE FOR JSON */
		const dict_values = {stock, time}
		const s = JSON.stringify(dict_values)
		/* IF EMPTY FIELD, GIVE ALERT AND RESTART */
		if (stock.length < 1 || time.length < 1)
		{
			alert("Field(s) are blank");
			return;
		}
		/* MAKE POST REQUEST CONTAINING INPUTTED VALUES (GO TO web.py LINE 28)*/
		$.ajax(
		{
			url:"/predict",
			type:"POST",
			contentType:"application/json",
			data: JSON.stringify(s)

		}).done(function(data){ /* GET RESPONSE VALUE (IMAGE NAME) FROM JSON REQUEST AND MAKE IMAGE FROM IT */
			let graph = document.createElement("img");
			graph.src = data;
			document.body.appendChild(graph);

		});
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
		document.getElementById("optionTwo").style.display = "none";

	}
	if (tab == 2)
	{
		document.getElementById("one").classList.remove("active");
		document.getElementById("two").classList.add("active");
		document.getElementById("optionOne").style.display = "none";
		document.getElementById("optionTwo").style.display = "inline-block";
	}
}