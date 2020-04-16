import * as React from "react";
import { connect } from "react-redux";
import libertem from '../../images/libertem_logo.png';
import { RootReducer } from "../../store";

const mapStateToProps = (state: RootReducer) => {
    return {
        noOfDatasets: state.datasets.ids.length,
        isVisible: state.openDataset.formVisible,
        isbusy: state.openDataset.busy,
    };
}

type MergedProps = ReturnType<typeof mapStateToProps>;

const TopComponent: React.SFC<MergedProps> = ({ noOfDatasets, isVisible, isbusy}) => {
    return(
        <div>
            { (!noOfDatasets && !isVisible && !isbusy) ?
            <div>
                <h1>Get Started With LiberTEM</h1>
                <img src={libertem} alt="libertem" style={{paddingTop: '5%', paddingBottom: '5%', opacity: '0.6'}}/>
                <div style={{marginLeft: '30%', marginRight: '30%', paddingBottom: '5%'}}>
                    <text>
                    LiberTEM is an open source platform for high-throughput distributed processing of pixelated scanning transmission electron microscopy (STEM) data. Click on "Browse" to select the dataset you want to analyze.
                    </text>
                </div>
            </div>
            : '' }
        </div>
    );
}

export default connect(mapStateToProps)(TopComponent);