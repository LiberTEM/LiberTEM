import * as React from "react";
import { Cell } from "./FileBrowserEntry";

const FileBrowserHeader: React.FC = () => {
    const alignRight: React.CSSProperties = {
        textAlign: "right",
    };

    // ugly hack: padding-right to compensate for scrollbar size
    return (
        <div style={{ paddingRight: "20px", paddingBottom: "10px" }}>
            <div style={{ display: "flex" }}>
                <div style={{ width: "20%", flexGrow: 1 }}>
                    <Cell>Name</Cell>
                </div>
                <div style={{ width: "10%", ...alignRight }}>
                    <Cell>Size</Cell>
                </div>
                <div style={{ width: "10%" }}>
                    <Cell>Owner</Cell>
                </div>
                <div style={{ width: "18%", ...alignRight }}>
                    <Cell>Created</Cell>
                </div>
                <div style={{ width: "18%", ...alignRight }}>
                    <Cell>Modified</Cell>
                </div>
            </div>
        </div>
    );
}

export default FileBrowserHeader;