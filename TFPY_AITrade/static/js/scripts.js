/**
 * 
 * @param {inputs} input The inputs that are going to be converted to a tag input
 * @param {list} whiteList the list of suggested options
 * @param {list} defaultTags options already in the input
 */
function inputTags(input, whiteList, defaultTags) {
    var tagify = new Tagify(input, {
        dropdown: {
            enabled: 0
        },
        whitelist: whiteList
    })

    tagify.on('change', console.log)

    tagify.addTags(defaultTags)
}

/**
 * 
 * @param {div} container The div containing the inputs that mark the date range
 * @param {select} selection The object selected in the select element
 * @param {input} iniRangeInput Input that corresponds with the begining of the range
 * @param {input} endRangeInput Input that corresponds with the end of the range
 * @returns 
 */
function dateSelector(container, selection, iniRangeInput, endRangeInput){
	value = 0;
	if(container.is(':hidden')){container.slideDown();}
	if (selection.index() == 1){value = 7;}
	else if (selection.index() == 2){value = 30;}
	else if (selection.index() == 3){value = 180;}
	else if (selection.index() == 4){container.slideUp();}
	else {return;}
	var today = new Date();
	var endRange = today.toISOString().split('T')[0];

	var yesterday = new Date(today.setDate(today.getDate() - value));
	var iniRange = yesterday.toISOString().split('T')[0];
	
	iniRangeInput.val(iniRange);
	endRangeInput.val(endRange);
}