export function formatHumanDate(seconds) {

    const getMinutes = (seconds) => {
        return (seconds / 60).toFixed(0)
    };

    // max unit is hours and min unit is minute
    let hours;
    let min;
    if (seconds > 3600) {
        hours = Math.floor((seconds / 3600));
        const remainSeconds = seconds - (hours * 3600);
        min = getMinutes(remainSeconds)
    } else {
        hours = null;
        min = getMinutes(seconds);
    }

    if (hours !== null) {
        return `${hours}h ${min}m`
    } else {
        return `${min}m`
    }
}

export function isEmpty(obj){
    return obj === null || obj === undefined;
}

export function formatFloat(num, length=4) {
    if(num !== undefined && num !== null){
        return num.toFixed(4)
    }else{
        return null;
    }
}
