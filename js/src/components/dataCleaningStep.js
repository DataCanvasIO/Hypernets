import {Card, Col, Row, Table, Tooltip} from "antd";
import {connect, Provider} from "react-redux";
import {createStore} from "redux";
import React from "react";
import {ConfigurationCard} from "./steps";
import {store} from "../index";



export function DataCleaningStep({data}) {


    const dataSource = data.extension?.unselected_features?.map((value, index, arr) => {
        return {
            key: index,
            feature_name: value.name,
            reason: value.reason, // fixme
        }
    });

    const columns = [
        {
            title: 'Feature name',
            dataIndex: 'feature_name',
            key: 'feature_name',
        },
        {
            title: 'Reason',
            dataIndex: 'reason',
            key: 'reason',
        }
    ];

    const title = <span>
        <Tooltip title={"sssssss"}>
            Data cleaning configuration
        </Tooltip>
    </span>;

    return <Row gutter={[4, 4]}>
        <Col span={10} >
            <Card title={title} bordered={false} style={{ width: '100%' }}>
                {
                    <ConfigurationCard configurationData={data.configuration.data_cleaner_params}/>
                }
            </Card>
        </Col>

        <Col span={10} offset={2} >
            <Card title="Removed features" bordered={false} style={{ width: '100%' }}>
                <Table dataSource={dataSource} columns={columns} />
            </Card>
        </Col>
    </Row>

}

// export const DataCleaningStepContainer = connect( state => {
//     // console.info("DataCleaningStepContainer state");
//     // console.info(state);
//     return {data: state.extension};
// }, dispatch => {
//     return {dispatch};
// })(DataCleaningStep);
//
// // Reducer
// function dataCleaningReducer(state = { count: 0 }, action) {
//     // Transform action to state
//     // console.info("dataCleaningReducer");
//     // console.info(action);
//     return action;
// }
//
// export const dataCleaningStore = createStore(dataCleaningReducer);


