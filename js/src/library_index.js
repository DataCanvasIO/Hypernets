import React from 'react';
import ReactDOM from 'react-dom';

import { Component  } from 'react'
import PropTypes from 'prop-types'

import { createStore } from 'redux'
import { Provider, connect } from 'react-redux'
import { ExperimentUI } from "./pages/experiment"



// Reducer
function counter(state = { count: 0 }, action) {
    const count = state.count
    switch (action.type) {
        case 'increase':
            return { count: action.value }
        default:
            return state
    }
}

// Store
export const store = createStore(counter)

export function renderPipelineMatrixBundle(ele){
    ReactDOM.render(
        // <MyComponent percentage={percentage} />,
        <Provider store={store}>
            <App />
        </Provider>,
        ele
    );
};


// Action
const increaseAction = { type: 'increase' }


// Map Redux state to component props
function mapStateToProps(state) {
    console.log("state:")
    console.log(state);
    return {
        percentage: parseInt(state.count)
    }
}

// Map Redux actions to component props
function mapDispatchToProps(dispatch) {
    return {
        onIncreaseClick: () => dispatch(increaseAction)
    }
}

// Connected Component
const App = connect(
    mapStateToProps,
    mapDispatchToProps
)(ExperimentUI);


// setInterval(function () {
//   store.dispatch(increaseAction);
// }, 1000);

