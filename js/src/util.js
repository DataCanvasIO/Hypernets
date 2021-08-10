import {notification} from "antd";
import uuidv4 from 'uuid/v4';


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
        if (min < 1){
            return `<1m`
        }else{
            return `${min}m`
        }
    }
}

export function isEmpty(obj){
    return obj === null || obj === undefined;
}

export function notEmpty(obj) {
    return obj !== undefined && obj !== null;
}

export function getOrDefault(obj, defaultValue) {
    if(obj !== undefined && obj !== null){
        return obj;
    }else{
        return defaultValue;
    }
}


export function formatFloat(num, length=4) {
    if(num !== undefined && num !== null){
        return num.toFixed(length)
    }else{
        return null;
    }
}

export const showNotification = (message) => {
    notification.error({
        key: uuidv4(),
        message: message,
        duration: 10,
    });
};
