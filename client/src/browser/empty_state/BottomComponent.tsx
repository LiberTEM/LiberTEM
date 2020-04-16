import * as React from "react";
import { connect } from "react-redux";
import github from '../../images/github_logo.png';
// import { RootReducer } from "../../store";
import EmptyStateProps from "./Props";

/*
const mapStateToProps = (state: RootReducer) => {
    return {
        noOfDatasets: state.datasets.ids.length,
        isVisible: state.openDataset.formVisible,
        isbusy: state.openDataset.busy,
    };
}
*/

const mapStateToProps = EmptyStateProps;

type MergedProps = ReturnType<typeof mapStateToProps>;

const BottomComponent: React.SFC<MergedProps> = ({ noOfDatasets, isVisible, isbusy}) => {
    return(
        <div>
            { (!noOfDatasets && !isVisible && !isbusy) ? 
                    <div style={{paddingTop: '5%', paddingBottom: '5%'}}>
                        <a href='https://libertem.github.io/LiberTEM/' target='_blank' rel='noopener'>Visit LiberTEM documentation</a>
                        <div style={{marginTop: '3%'}}>
                            <img src={github} alt="github" style={{height: '25px', width: '25px', marginLeft: '-25px'}}/>
                            <a href='https://github.com/LiberTEM/LiberTEM' target='_blank' rel='noopener'>Star us on GitHub</a>
                        </div>
                    </div>
                    : '' }
        </div>
    );
}

export default connect(mapStateToProps)(BottomComponent);