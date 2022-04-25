import * as React from "react";
import { useEffect, useState } from "react";
import { useDispatch, useSelector } from "react-redux";
import { Dispatch } from "redux";
import { Button, Dropdown, DropdownProps, Header, Icon, Modal, Segment, Tab } from "semantic-ui-react";
import { AllActions } from "../../actions";
import { AnalysisState } from "../../analysis/types";
import { dispatchGenericError } from "../../errors/helpers";
import { writeClipboard } from "../../helpers";
import { getApiBasePath } from "../../helpers/apiHelpers";
import { JobStatus } from "../../job/types";
import { CopyAnalysis } from "../../messages";
import { RootReducer } from "../../store";
import { getNotebook } from '../api';
import { getMetadata } from "../getMetadata";
import { CompoundAnalysisState } from "../types";

interface DownloadItemsProps {
    compoundAnalysis: CompoundAnalysisState,
    currentFormat: string,
}

const DownloadItems: React.FC<DownloadItemsProps> = ({
    compoundAnalysis, currentFormat
}) => {

    const basePath = getApiBasePath();
    const downloadUrl = (analysisId: string) => (
        `${basePath}compoundAnalyses/${compoundAnalysis.compoundAnalysis}/analyses/${analysisId}/download/${currentFormat}/`
    )

    const analysesById = useSelector((state: RootReducer) => state.analyses.byId);
    const jobsById = useSelector((state: RootReducer) => state.jobs.byId);

    const analyses = compoundAnalysis.details.analyses.map(analysis => analysesById[analysis]).filter(analysis =>
        analysis.jobs.some(jobId => jobsById[jobId].status === JobStatus.SUCCESS)
    );

    const getAnalysisDescription = (analysis: AnalysisState) => getMetadata(analysis.details.analysisType).desc;

    const getDownloadChannels = (analysis: AnalysisState) => {
        if (!analysis.displayedJob) {
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
            {analyses.map((analysis) => (
                <li key={analysis.id}>
                    <a href={downloadUrl(analysis.id)}>
                        {getAnalysisDescription(analysis)} (channels: {getDownloadChannels(analysis).join(", ")})
                        </a>
                </li>
            ))}
        </ul>
    )
}

interface CopyScriptsProps {
    compoundAnalysis: CompoundAnalysisState,
}

const CopyScripts: React.FC<CopyScriptsProps> = ({ compoundAnalysis }) => {
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

    const dispatch: Dispatch<AllActions> = useDispatch();

    const cell = (code: string) => {
        const copy = () => {
            writeClipboard(code, dispatch);
        };

        return (
            <Segment padded>
                <Button floated={"right"} icon={"copy"} onClick={copy} />
                <pre>{code}</pre>
            </Segment>
        );
    };

    const copyCompleteNotebook = () => {
        const firstPart = [notebook.dependency, notebook.initial_setup, notebook.ctx, notebook.dataset].join("\n\n");
        const joinCode = (analysis: CopyAnalysis) => `${analysis.analysis}\n${analysis.plot.join("\n\n")}`
        const secondPart = notebook.analysis.map(joinCode).join("\n\n");
        writeClipboard(`${firstPart}\n\n${secondPart}`, dispatch);
    };

    useEffect(() => {
        getNotebook(compoundAnalysis.compoundAnalysis).then(CurrentNotebook => {
            setNotebook({
                dependency: CurrentNotebook.dependency,
                initial_setup: CurrentNotebook.initial_setup,
                ctx: CurrentNotebook.ctx,
                dataset: CurrentNotebook.dataset,
                analysis: CurrentNotebook.analysis,
            });
        }).catch(() => dispatchGenericError("could not get notebook", dispatch))
    }, [compoundAnalysis.compoundAnalysis]);

    return (
        <>
            <Segment clearing>
                <Header floated={"left"}>Notebook</Header>
                <Button icon labelPosition="left" floated={"right"} onClick={copyCompleteNotebook}>
                    <Icon name="copy" />
                    Complete notebook
                </Button>
            </Segment>
            <Modal.Content scrolling>
                {[notebook.dependency, notebook.initial_setup, notebook.ctx, notebook.dataset].map(cell)}
                {notebook.analysis.map(analysis => (
                    <>
                        {cell(analysis.analysis)}
                        {analysis.plot.map(cell)}
                    </>
                ))}
            </Modal.Content>
        </>
    );
};

interface DownloadScriptsProps {
    compoundAnalysis: CompoundAnalysisState,
}

const DownloadScripts: React.FC<DownloadScriptsProps> = ({ compoundAnalysis }) => {
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
    value: string;
}>;

interface DownloadResultItemProps {
    formatOptions: FormatOptions,
    onFormatChange: (e: React.SyntheticEvent, data: DropdownProps) => void,
    currentFormat: string,
    compoundAnalysis: CompoundAnalysisState,
}

const DownloadResultItem: React.FC<DownloadResultItemProps> = ({
    formatOptions, onFormatChange, currentFormat, compoundAnalysis,
}) => (
    <Tab.Pane>
        <Header >
            Download Results, format: <Dropdown inline options={formatOptions} onChange={onFormatChange} value={currentFormat} />
        </Header>
        <Header as="h3">Available results:</Header>
        <DownloadItems compoundAnalysis={compoundAnalysis} currentFormat={currentFormat} />
    </Tab.Pane>
);

interface DownloadNotebookItemProps {
    compoundAnalysis: CompoundAnalysisState,
}

const DownloadNotebookItem: React.FC<DownloadNotebookItemProps> = ({
    compoundAnalysis
}) => (
    <Tab.Pane>
        <Header as="h3">Available scripts: </Header>
        <DownloadScripts compoundAnalysis={compoundAnalysis} />
    </Tab.Pane>
);

interface CopyNotebookItemProps {
    compoundAnalysis: CompoundAnalysisState,
}

const CopyNotebookItem: React.FC<CopyNotebookItemProps> = ({
    compoundAnalysis,
}) => (
    <Tab.Pane>
        <CopyScripts compoundAnalysis={compoundAnalysis} />
    </Tab.Pane>
);

const Download: React.FC<DownloadProps> = ({ compoundAnalysis }) => {
    const formats = useSelector((state: RootReducer) => state.config.resultFileFormats);
    const formatOptions: FormatOptions = Object.keys(formats).map(identifier => ({
        value: identifier,
        text: formats[identifier].description,
    }));

    const [currentFormat, setFormat] = useState(formatOptions[0]?.value);

    // we may be called before the config is completely loaded, so we
    // need to set the format after the list of formats is available
    React.useEffect(() => {
        if (formatOptions.length !== 0 && !currentFormat) {
            setFormat(formatOptions[0].value);
        }
    }, [formatOptions, currentFormat])

    const onFormatChange = (e: React.SyntheticEvent, data: DropdownProps) => {
        if(data.value) {
            setFormat(data.value.toString());
        }
    }

    const panes = [
        {
            menuItem: "Download result",
            render: () => <DownloadResultItem formatOptions={formatOptions} onFormatChange={onFormatChange} currentFormat={currentFormat} compoundAnalysis={compoundAnalysis} />
        },
        {
            menuItem: "Download notebook",
            render: () => <DownloadNotebookItem compoundAnalysis={compoundAnalysis} />,
        },
        {
            menuItem: "Copy notebook",
            render: () => <CopyNotebookItem compoundAnalysis={compoundAnalysis} />,
        },
    ];


    return (
        <Modal trigger={
            <Button icon>
                <Icon name='download' />
                Download
            </Button>
        }>
            <Tab panes={panes} />
        </Modal>
    );
}

export default Download;
