import * as React from "react";
import { useState } from "react";
import { useSelector } from "react-redux";
import { Button, Dropdown, DropdownProps, Header, Icon, Modal, Popup } from "semantic-ui-react";
import { AnalysisState } from "../../analysis/types";
import { getApiBasePath } from "../../helpers/apiHelpers";
import { JobStatus } from "../../job/types";
import { RootReducer } from "../../store";
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
        </Modal>
    );
}

export default Download;