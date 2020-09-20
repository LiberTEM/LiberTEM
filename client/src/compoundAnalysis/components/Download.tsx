import * as React from "react";
import { useEffect, useState } from "react";
import { useSelector } from "react-redux";
import { Button, Dropdown, DropdownProps, Header, Icon, Modal, Segment, Tab} from "semantic-ui-react";
import { AnalysisState } from "../../analysis/types";
import { getApiBasePath } from "../../helpers/apiHelpers";
import { JobStatus } from "../../job/types";
import { CopyAnalysis } from "../../messages";
import { RootReducer } from "../../store";
import { getNotebook } from '../api'
import { getMetadata } from "../getMetadata";
import { CompoundAnalysisState } from "../types";

interface DownloadItemsProps {
    compoundAnalysis: CompoundAnalysisState,
    currentFormat?: string,
}

const DownloadItems: React.SFC<DownloadItemsProps> = ({
    compoundAnalysis, currentFormat
}) => {

    const basePath = getApiBasePath();
    const downloadUrl = (analysisId: string) => (
        `${basePath}compoundAnalyses/${compoundAnalysis.compoundAnalysis}/analyses/${analysisId}/download/${currentFormat}/`
    )

    const analysesById = useSelector((state: RootReducer) => {
        return state.analyses.byId;
    });

    const jobsById = useSelector((state: RootReducer) => {
        return state.jobs.byId;
    });

    const analyses = compoundAnalysis.details.analyses.map(analysis => {
        return analysesById[analysis];
    }).filter(analysis => {
        return analysis.jobs.some(jobId => jobsById[jobId].status === JobStatus.SUCCESS);
    })

    const getAnalysisDescription = (analysis: AnalysisState) => {
        return getMetadata(analysis.details.analysisType).desc;
    }

    const getDownloadChannels = (analysis: AnalysisState) => {
        if(!analysis.displayedJob) {
            return [];
        }
        return jobsById[analysis.displayedJob].results.filter(
            result => result.description.includeInDownload
        ).map(
            result => result.description.title
        )
    }

    return (
        <ul>
            {analyses.map((analysis) => {
                return (
                    <li key={analysis.id}>
                        <a href={downloadUrl(analysis.id)}>
                            {getAnalysisDescription(analysis)} (channels: {getDownloadChannels(analysis).join(", ")})
                        </a>
                    </li>
                );
            })}
        </ul>
    )
}

interface DownloadItemsProps {
    compoundAnalysis: CompoundAnalysisState,
}

const CopyScripts: React.SFC<DownloadItemsProps> = ({ compoundAnalysis }) => {
    const initialAnalysis: CopyAnalysis[] = [
        {
            analysis: "",
            plot: [""],
        },
    ];

    const [notebook, setNotebook] = useState({
        dependency: "",
        initial_setup: "",
        ctx: "",
        dataset: "",
        analysis: initialAnalysis,
    });

    const cell = (code: string) => {
        const copy = () => {
            navigator.clipboard.writeText(code);
        };

        return (
            <Segment padded={true}>
                <Button floated={"right"} icon={"copy"} onClick={copy} />
                <pre>{code}</pre>
            </Segment>
        );
    };

    const copyCompleteNotebook = () => {
        const firstPart = [notebook.dependency, notebook.initial_setup, notebook.ctx, notebook.dataset].join("\n\n");
        const joinCode = (analysis: CopyAnalysis) => {
                return `${analysis.analysis}\n${analysis.plot.join("\n\n")}`
        }
        const secondPart = notebook.analysis.map(joinCode).join("\n\n");
        navigator.clipboard.writeText(`${firstPart}\n\n${secondPart}`);
    };

    useEffect(() => {
        const copyNotebook = async () => {
            await getNotebook(compoundAnalysis.compoundAnalysis).then(CurrentNotebook => {
                setNotebook({
                    dependency: CurrentNotebook.dependency,
                    initial_setup: CurrentNotebook.initial_setup,
                    ctx: CurrentNotebook.ctx,
                    dataset: CurrentNotebook.dataset,
                    analysis: CurrentNotebook.analysis,
                });
            });
        };
        copyNotebook();
    }, [compoundAnalysis.compoundAnalysis]);

    return (
        <>
            <Segment clearing={true}>
                <Header floated={"left"}>Notebook</Header>
                <Button icon={true} labelPosition="left" floated={"right"} onClick={copyCompleteNotebook}>
                    <Icon name="copy" />
                    Complete notebook
                </Button>
            </Segment>
            <Modal.Content scrolling={true}>
                {[notebook.dependency, notebook.initial_setup, notebook.ctx, notebook.dataset].map(cell)}
                {notebook.analysis.map(analysis => {
                    return (
                        <>
                            {cell(analysis.analysis)}
                            {analysis.plot.map(cell)}
                        </>
                    );
                })}
            </Modal.Content>
        </>
    );
}; 


const DownloadScripts: React.SFC<DownloadItemsProps> = ({ compoundAnalysis }) => {
    const basePath = getApiBasePath();
    const downloadUrl = `${basePath}compoundAnalyses/${compoundAnalysis.compoundAnalysis}/download/notebook/`;

    return (
        <ul>
            <li>
                <a href={downloadUrl}>notebook corresponding to analysis</a>
            </li>
        </ul>
    );
};

interface DownloadProps {
    compoundAnalysis: CompoundAnalysisState,
}

type FormatOptions = Array<{
    text: string;
    value: any;
}>;

const Download: React.SFC<DownloadProps> = ({ compoundAnalysis }) => {
    const formats = useSelector((state: RootReducer) => state.config.resultFileFormats);
    const formatOptions: FormatOptions = Object.keys(formats).map(identifier => {
        return {
            value: identifier,
            text: formats[identifier].description,
        }
    });

    const [currentFormat, setFormat] = useState(formatOptions[0]?.value);

    // we may be called before the config is completely loaded, so we
    // need to set the format after the list of formats is available
    React.useEffect(() => {
        if(formatOptions.length !== 0 && !currentFormat) {
            setFormat(formatOptions[0].value);
        }
    }, [formatOptions, currentFormat])

    const onFormatChange = (e: React.SyntheticEvent, data: DropdownProps) => {
        setFormat(data.value);
    }

    const panes = [
        {
            menuItem: "Download result",
            render: () => (
                <Tab.Pane>
                    <Header >
                        Download Results, format: <Dropdown inline={true} options={formatOptions} onChange={onFormatChange} value={currentFormat} />
                    </Header>
                    <Header as="h3">Available results:</Header>
                    <DownloadItems compoundAnalysis={compoundAnalysis} currentFormat={currentFormat} />
                </Tab.Pane>
            ),
        },
        {
            menuItem: "Download notebook",
            render: () => (
                <Tab.Pane>
                    <Header as="h3">Available scripts: </Header>
                    <DownloadScripts compoundAnalysis={compoundAnalysis} />
                </Tab.Pane>
            ),
        },
        {
            menuItem: "Copy notebook",
            render: () => (
                <Tab.Pane>
                    <CopyScripts compoundAnalysis={compoundAnalysis} />
                </Tab.Pane>
            ),
        },
    ];
      

    return (
        <Modal trigger={
            <Button icon={true}>
                <Icon name='download' />
                Download
            </Button>
        }>
            <Tab panes={panes} />
        </Modal>
    );
}

export default Download;