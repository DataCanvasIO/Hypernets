export function formatHumanDate(seconds) {
    // max unit is hours and min unit is minute
    let hours;
    if (seconds > 3600) {
        hours = Math.floor((seconds / 3600))
    } else {
        hours = null;
    }

    const min = Math.floor((seconds % 3600) / 60);

    if (hours !== null) {
        return `${hours}h ${min}m`
    } else {
        return `${min}m`
    }
}

export function isEmpty(obj){
    return obj === null || obj === undefined;
}
