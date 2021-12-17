import * as React from "react";
import { useDispatch, useSelector } from "react-redux";
import { Button, Icon, IconProps, Segment } from "semantic-ui-react";
import { RootReducer } from "../../store";
import * as analysisActions from "../actions";
import { getAnalysisStatus } from "../helpers";
import { CompoundAnalysisState } from "../types";
import Download from "./Download";

interface ToolbarProps {
    compoundAnalysis: CompoundAnalysisState,
    busyIdxs: number[],
    extra?: React.ReactNode,
    onApply: () => void,
}


type MergedProps = ToolbarProps;

const Toolbar: React.FC<MergedProps> = ({ busyIdxs, onApply, compoundAnalysis, extra }) => {
    const dispatch = useDispatch();
    const handleRemove = () => dispatch(analysisActions.Actions.remove(compoundAnalysis.compoundAnalysis));
    const analyses = useSelector((state: RootReducer) => state.analyses);
    const jobs = useSelector((state: RootReducer) => state.jobs);
    const status = getAnalysisStatus(
        compoundAnalysis, analyses, jobs,
        busyIdxs
    );
    const running = status === "busy";
    const applyIconProps: IconProps = running ? { name: 'cog', loading: true } : { name: 'check' }

    return (
        <Segment attached="bottom">
            <Button.Group>
                <Button primary onClick={onApply} icon>
                    <Icon {...applyIconProps} />
                    Apply
                </Button>
                <Download compoundAnalysis={compoundAnalysis} />
                <Button onClick={handleRemove} icon>
                    <Icon name='remove' />
                    Remove
                </Button>
                {extra}
            </Button.Group>
        </Segment>
    );
}

export default Toolbar;
