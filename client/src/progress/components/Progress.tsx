import React from "react";
import { RootReducer } from "../../store";
import { connect } from "react-redux";
import { getTotalProgress } from "../helpers";
import { ProgressDetails } from "../../messages";
import { Progress as ProgressBar, Transition } from "semantic-ui-react";

const mapStateToProps = (state: RootReducer) => ({
    progress: state.progress,
})

type MergedProps = ReturnType<typeof mapStateToProps>;

const progressAsFract = (prog: ProgressDetails): number => {
    return prog.numFramesComplete / prog.numFrames;
}

const Progress: React.FC<MergedProps> = ({ progress }) => {
    const totalProgres = getTotalProgress(progress);
    const prog = progressAsFract(totalProgres);
    const cleanProg = isNaN(prog) ? 1 : prog;
    const done = cleanProg >= 1;
    const color = done ? 'green' : 'red';

    return (
        <Transition animation="fade" duration={500} visible={!done}>
            <div style={{ flexGrow: 1 }}>
                <ProgressBar color={color} indicating={!done} percent={cleanProg * 100} style={{ marginBottom: 0 }} />
            </div>
        </Transition>
    );
}

export default connect(mapStateToProps)(Progress);