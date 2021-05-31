import * as React from "react";
import './exploreDataset.css';

// JSON
// React

function formatMessage({id}) {
    return id;
}

function FeatureBar({first, latest, color, nFeatures, totalFeatures, typeName}) {
    //
    const numberPercent = ((nFeatures / totalFeatures) * 100).toFixed(0);
    let borderRadius ;
    if(first){
        if(latest){
            borderRadius = {borderRadius: '20px 20px 20px 20px'};
        }else{
            borderRadius = {borderRadius: '20px 0 0 20px'};
        }
    }else{
        if(latest){
            borderRadius = {borderRadius: '0px 20px 20px 0'};
        }else{
            borderRadius = {}
        }
    }

    return <div  style={{ width: `${numberPercent}`, backgroundColor: color, ...borderRadius}}>
            {nFeatures} {typeName} ({numberPercent})
    </div>
}

function FeatureLegend() {
    // return <div className={'number'}>
    //     <span className={'numberCir'}/>
    //     <span className={'symbol'}> {`${formatMessage({id: 'explore.num'})}`} </span>
    //     <span className={'comment'}> {number} {`${formatMessage({id: 'explore.cols'})}`}，{numberPercent} </span>
    // </div>
}

export function Dataset({data}) {

    const { nContinuous, nCategoricalCols, nDatetimeCols, nTextCols, nOthers } = data.featureDistribution;
    const total = nContinuous + nCategoricalCols + nDatetimeCols + nTextCols + nOthers;

    const barList = [
        {
            typeName: "Continuous",
            color: "rgb(0, 183, 182)",
            nFeatures: nContinuous
        },
        {
            typeName: "Categorical",
            color: "rgb(0, 156, 234)",
            nFeatures: nCategoricalCols
        },
        {
            typeName: "Datetime",
            color: "rgb(244, 148, 49)",
            nFeatures: nDatetimeCols
        },
        {
            typeName: "Text",
            color: "rgb(125, 0, 249)",
            nFeatures: nTextCols
        },
        {
            typeName: "Other",
            color: "rgb(105, 125, 149)",
            nFeatures: nOthers
        }
    ];
    // 数据不能为空，
    // find first not empty
    var firstNonEmpty  = null ;
    var latestNonEmpty  = null ;
    for (var i = 0 ; i < barList.length ; i++){
        const item = barList[i];
        if(firstNonEmpty === null){
            if(item.nFeatures > 0){
                firstNonEmpty = i;
            }
        }
        const backIndex = barList.length - i - 1;
        const backItem = barList[backIndex];
        if(latestNonEmpty === null){
            if(backItem.nFeatures > 0){
                latestNonEmpty = backIndex;
            }
        }
    }

    if(firstNonEmpty !== null && latestNonEmpty !== null){
        const bars = barList.map((value, index , array) => {
            return <FeatureBar
                    typeName={value.typeName}
                    first={index === firstNonEmpty}
                    latest={index === latestNonEmpty}
                    color={value.color}
                    nFeatures={value.nFeatures}
                    totalFeatures={total}
            />
        });

        return <>
            <div className={'bar'}>
                {bars}
            </div>

            {/*<div className={'legend'}>*/}

            {/*    <div className={'type'}>*/}
            {/*        <span className={'typeCir'}/>*/}
            {/*        <span className={'symbol'}>{`${formatMessage({id: 'explore.category'})}`}</span>*/}
            {/*        <span className={'comment'}>{category} {`${formatMessage({id: 'explore.cols'})}`}，{typePercent}</span>*/}
            {/*    </div>*/}
            {/*    <div className={date}>*/}
            {/*        <span className={'dateCir'}/>*/}
            {/*        <span className={'symbol'}>{`${formatMessage({id: 'explore.date'})}`}</span>*/}
            {/*        <span className={'comment'}>{date} {`${formatMessage({id: 'explore.cols'})}`}，{datePercent}</span>*/}
            {/*    </div>*/}
            {/*    <div className={text}>*/}
            {/*        <span className={'textCir'}/>*/}
            {/*        <span className={'symbol'}>{`${formatMessage({id: 'explore.text'})}`}</span>*/}
            {/*        <span className={'comment'}>{text} {`${formatMessage({id: 'explore.cols'})}`}，{textPercent}</span>*/}
            {/*    </div>*/}
            {/*</div>*/}
        </>



    }else{
        return <span>
            Error, features is empty.
        </span>
    }



}