import * as React from "react";
import { connect } from "react-redux";
import { Header, Icon, Message, Modal, Popup, Segment } from 'semantic-ui-react';
import AnalysisList from "../../analysis/components/AnalysisList";
import { AnalysisState } from "../../analysis/types";
import { filterWithPred } from "../../helpers/reducerHelpers";
import { DatasetState, DatasetStatus } from "../../messages";
import { RootReducer } from "../../store";
import AddAnalysis from "./AddAnalysis";
import DatasetInfo from "./DatasetInfo";
import DatasetToolbar from "./DatasetToolbar";

interface DatasetProps {
    dataset: DatasetState
}

const mapStateToProps = (state: RootReducer, ownProps: DatasetProps) => {
    const p = (analysis: AnalysisState) => analysis.dataset === ownProps.dataset.id;
    return {
        analyses: filterWithPred(state.analyses, p),
    }
}

type MergedProps = DatasetProps & ReturnType<typeof mapStateToProps>;

const DatasetComponent: React.SFC<MergedProps> = ({ dataset, analyses }) => {
    const msg = {
        [DatasetStatus.OPENING]: `Opening dataset ${dataset.params.name}`,
        [DatasetStatus.DELETING]: `Closing dataset ${dataset.params.name}`,
    }
    if (dataset.status === DatasetStatus.OPENING || dataset.status === DatasetStatus.DELETING) {
        return (
            <>
                <Header as="h2" dividing={true}>{dataset.params.name}</Header>
                <Message icon={true}>
                    <Icon name='cog' loading={true} />
                    <Message.Content>
                        <Message.Header>{msg[dataset.status]}</Message.Header>
                    </Message.Content>
                </Message>
            </>
        );
    }

    return (
        <Segment.Group style={{ marginTop: "3em", marginBottom: "3em" }}>
            <Segment.Group horizontal={true}>
                <Segment>
                    <Header as="h2">
                        <Icon name="database" />
                        <Modal trigger={
                            <Header.Content>
                                {dataset.params.name}
                                {' '}
                                <Icon name="info circle" size="small" link={true} />
                            </Header.Content>
                        }>
                            <Popup.Header>{dataset.params.type} Dataset {dataset.params.name}</Popup.Header>
                            <Popup.Content>
                                <DatasetInfo dataset={dataset} />
                            </Popup.Content>
                        </Modal>
                    </Header>
                </Segment>
                <Segment style={{ flexShrink: 1, flexGrow: 0 }}>
                    <DatasetToolbar dataset={dataset} />
                </Segment>
            </Segment.Group>
            {
                analyses.ids.length > 0 ? (
                    <>
                        <Segment>
                            <AnalysisList analyses={analyses} />
                        </Segment>
                    </>
                ) : null
            }
            <Segment textAlign="center">
                <AddAnalysis dataset={dataset} />
            </Segment>
        </Segment.Group>
    );
}

const DatasetContainer = connect(mapStateToProps)(DatasetComponent);

export default DatasetContainer;