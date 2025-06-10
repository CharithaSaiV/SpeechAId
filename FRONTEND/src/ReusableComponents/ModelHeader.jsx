import React from "react";
import { Box, Tooltip, Typography } from "@mui/material";
import IconButton from "@mui/material/IconButton";
import { Icon } from "@iconify/react";

const ModalHeader = ({ heading, handleClose, handleDownload }) => {
  return (
    <div style={{ position: "sticky", top: 0, zIndex: 1000 }}>
      <Box
        height="30px"
        backgroundColor="#3B82F6"
        display="flex"
        alignItems="center"
        justifyContent="center"
      >
        <Typography
          variant="h6"
          style={{
            color: "#FFFFFF",
            width: "100%",
            textAlign: "center",
            fontWeight: 600,
          }}
        >
          {heading}
        </Typography>
        {handleDownload && (
          <IconButton
            edge="start"
            color="white"
            aria-label="close"
            onClick={handleDownload}
            style={{
              float: "right",
              display: "flex",
              alignItems: "flex-end",
              justifyContent: "flex-end",
              marginRight: "12px",
              right: "10",
            }}
          >
            <Tooltip title="Download Offer">
              <Icon
                icon="material-symbols-light:download-2"
                sx={{ color: "white", fontWeight: 700 }}
              />
            </Tooltip>
          </IconButton>
        )}
        {handleClose && (
          <Icon
            icon="zondicons:close-outline"
            style={{
              float: "right",
              display: "flex",
              alignItems: "flex-end",
              justifyContent: "flex-end",
              marginRight: "12px",
              cursor: "pointer",
              color: "white",
              width: "20px",
              height: "20px",
            }}
            onClick={() => {
              handleClose();
            }}
          />
        )}
      </Box>
    </div>
  );
};

export default ModalHeader;
