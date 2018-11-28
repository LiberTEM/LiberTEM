import * as React from "react";

const FileBrowserHeader: React.SFC = () => {
    const tsStyles: React.CSSProperties = {
        textAlign: "right",
    };

    // ugly hack: padding-right to compensate for scrollbar size
    return (
        <div style={{ paddingRight: "20px", paddingBottom: "10px" }}>
            <div style={{ display: "flex" }}>
                <div style={{ width: "20%", flexGrow: 1 }}>
                    Name
                </div>
                <div style={{ width: "15%" }}>Owner</div>
                <div style={{ width: "18%", ...tsStyles }}>Created</div>
                <div style={{ width: "18%", ...tsStyles }}>Modified</div>
            </div>
        </div>
    );
}

export default FileBrowserHeader;