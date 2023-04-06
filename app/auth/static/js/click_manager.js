var unameN = 1, emailN = 1, blistN = 1;

function counter(element, type) {
    let n = type + 'N';
    if (globalThis[n] < 3) {
        globalThis[n]++;
        globalThis[type] = element.innerHTML;
    } else {
        var input_e = document.createElement('input');
        input_e.setAttribute('name', type);
        input_e.setAttribute('class', 'userInput');
        input_e.setAttribute('value', globalThis[type]);
        input_e.setAttribute('style', 'text-align:center;');
        element.parentNode.replaceChild(input_e, element);
        exceptThis(input_e);
        globalThis[n] = 0;
    }
}


function unameCounter(element) {
    counter(element, 'uname');
}

function emailCounter(element) {
    counter(element, 'email');
}

function blistCounter(element) {
    counter(element, 'blist');
}


function hasChanged(name) {
    var originalVal = globalThis[name];
    var inputElement = document.getElementsByName(name)[0];
    if (!inputElement)
        return false;
    return originalVal !== inputElement.value;
}

function submitForm(token) {
    if (hasChanged('uname') || hasChanged('email') || hasChanged('blist')) {
        document.getElementById('update').submit();
    }
}


function exceptThis(element) {
    element.addEventListener('click', function (event) {
        event.stopPropagation();
    })
};

    
// quick tour
// https://www.youtube.com/watch?v=mV4jKBrIHvk
// thunkable
// https://x.thunkable.com/projects/61dcd06603e236033cc4870e/237f6704-2498-40f4-8478-cdc367207199/designer
