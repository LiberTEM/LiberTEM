import * as React from "react";
import { useState } from "react";
import { useSelector } from "react-redux";
import { Button, Dropdown, DropdownProps, Header, Icon, Modal, Popup, Segment} from "semantic-ui-react";
import { AnalysisState } from "../../analysis/types";
import { getApiBasePath } from "../../helpers/apiHelpers";
import { JobStatus } from "../../job/types";
import { RootReducer } from "../../store";
import { getMetadata } from "../getMetadata";
import { CompoundAnalysisState } from "../types";
import { getNotebook } from '../api'
 import { CopyAnalysis, CopyNotebookResponse } from "../../messages";

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

const CopyScripts: React.SFC<DownloadItemsProps> = ({compoundAnalysis}) => {

    const [openModal, setOpen] = useState(false)

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

    const copyNotebook = async () => {
        await getNotebook(compoundAnalysis.compoundAnalysis).then(notebook => {
            setNotebook({
                dependency: notebook.dependency,
                initial_setup: notebook.initial_setup,
                ctx: notebook.ctx,
                dataset: notebook.dataset,
                analysis: notebook.analysis,
            });
            setOpen(true);
        });
    };

    const cell = (code:string) => {
        return (
            <Segment padded>
                <Button
                    floated={"right"}
                    icon={"copy"}
                    onClick={() => {
                        navigator.clipboard.writeText(code);
                    }}
                />
                <p>
                    {code.split("\n").map((item, i) => {
                        return <p key={i}>{item}</p>;
                    })}
                </p>
            </Segment>
        );
    }

    const copyCompleteNotebook = (notebook: CopyNotebookResponse) => {
        const firstPart = [notebook.dependency, notebook.initial_setup, notebook.ctx, notebook.dataset].join("\n")
        const secondPart = notebook.analysis.map(analysis => `${analysis.analysis}\n${analysis.plot.join("\n")}`)
        navigator.clipboard.writeText(`${firstPart}\n${secondPart}`)
    }

    return (
        <Modal
            trigger={<Button onClick={copyNotebook}>Copy Notebook</Button>}
            open={openModal}
            onClose={() => {
                setOpen(false);
            }}
        >
            <Modal.Header>
                Notebook 
                <Button icon labelPosition='left' floated={"right"} onClick={() => copyCompleteNotebook(notebook)}>
                    <Icon name="copy" />
                    Complete notebook
                </Button>
            </Modal.Header>
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
        </Modal>
    );
} 


const DownloadScripts: React.SFC<DownloadItemsProps> = ({compoundAnalysis}) => {

    const basePath = getApiBasePath();
    const downloadUrl = `${basePath}compoundAnalyses/${compoundAnalysis.compoundAnalysis}/download/notebook/`

    
    if (compoundAnalysis[`details`][`mainType`] === 'CLUST') {
        return(
            <ul>
                <li>
                    Under Development
                </li>
            </ul>
        )
        } else {
            return(
                <ul>
                    <li>
                        <a href={downloadUrl}>
                            notebook corresponding to analysis
                        </a>
                    </li>
                    <li>
                        <CopyScripts compoundAnalysis={compoundAnalysis}/>
                    </li>
                </ul>
            )
        }
}

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

    return (
        <Modal trigger={
            <Button icon={true}>
                <Icon name='download' />
                Download
            </Button>
        }>
            <Popup.Header>
                Download Results, format: <Dropdown inline={true} options={formatOptions} onChange={onFormatChange} value={currentFormat} />
            </Popup.Header>
            <Popup.Content>
                <Header as="h3">Available results:</Header>
                <DownloadItems compoundAnalysis={compoundAnalysis} currentFormat={currentFormat} />
            </Popup.Content>
            <Popup.Content>
                <Header as="h3">Available scripts: </Header>
                <DownloadScripts compoundAnalysis={compoundAnalysis} />
            </Popup.Content>
        </Modal>
    );
}

export default Download;
