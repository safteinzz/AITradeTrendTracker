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